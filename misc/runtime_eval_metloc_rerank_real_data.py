"""
Runtime evaluation for HOTFormerMetricLoc (with re-ranking) on real data (validation set).
"""
import argparse
import os
import time

import tqdm
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from dataset.dataset_utils import make_dataloaders
from eval.sgv.sgv_utils import sgv_parallel
from eval.egonn_utils import ransac_fn
from models.model_factory import model_factory
from models.egonn import MinkGL as EgoNN
from models.hotformerloc_metric_loc import HOTFormerMetricLoc
from misc.torch_utils import set_seed, release_cuda, to_device
from misc.utils import TrainingParams

WARMUP_ITERS = 10
RUN_ITERS = 3


def setup_model():
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.model_params.return_feats_and_attn_maps = False  # saves mem + runtime
    # params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    model.print_info()

    model.to(device)
    model.eval()
    if isinstance(model, HOTFormerMetricLoc):
        model.benchmark()

    if args.use_sgv:
        params.model_params.rerank_mode = 'sgv'

    # Get val dataloader
    params.val_batch_size = args.num_neighbours
    params.local.batch_size = 1
    params.num_workers = 4
    dataloaders = make_dataloaders(params, local=True, validation=True)
    # dataloader = dataloaders['local_val']

    return params, device, model, dataloaders

def main():
    params, device, model, dataloaders = setup_model()

    if isinstance(model, EgoNN):
        num_points, global_timings, metric_loc_timings, rerank_timings = (
            runtime_eval_egonn(params, device, model, dataloaders)
        )
    else:
        num_points, global_timings, metric_loc_timings, rerank_timings = (
            runtime_eval_hotfloc(params, device, model, dataloaders)
        )

    num_points = np.asarray(num_points)
    print(f"Pointcloud size: mean - {num_points.mean():.0f} pts, std - {num_points.std():.0f} pts")
    if len(global_timings) > 0:
        global_timings = np.asarray(global_timings)
        print(f"RUNTIME (global descriptor): mean - {global_timings.mean():.2f}ms, std - {global_timings.std():.2f}ms")
    metric_loc_timings = np.asarray(metric_loc_timings)
    print(f"RUNTIME (metric loc): mean - {metric_loc_timings.mean():.2f}ms, std - {metric_loc_timings.std():.2f}ms")
    rerank_timings = np.asarray(rerank_timings)
    print(f"RUNTIME (reranking): mean - {rerank_timings.mean():.2f}ms, std - {rerank_timings.std():.2f}ms")

    print(f"Max mem allocated {torch.cuda.max_memory_allocated(device=None) / (1024 ** 2):.2f} MB memory")
    # print(f"mem allocated {torch.cuda.memory_allocated(device=None) / (1024 ** 2):.2f} MB memory")
    torch.cuda.reset_peak_memory_stats(device=None)

def runtime_eval_hotfloc(params, device, model, dataloaders):
    # starter = torch.cuda.Event(enable_timing=True)
    # ender = torch.cuda.Event(enable_timing=True)
    global_timings = []
    metric_loc_timings = []
    rerank_timings = []
    num_points = []

    # PR + Metric Loc loop
    for ii, local_batch in tqdm.tqdm(enumerate(dataloaders['local_val']), total=len(dataloaders['local_val'])):
        if ii == args.max_samples:
            break
        num_points.append(local_batch['anc_batch']['points'].npt)
        num_points.append(local_batch['pos_batch']['points'].npt)
        local_batch = to_device(local_batch, device, non_blocking=True, construct_octree_neigh=True)
        with torch.inference_mode():
            # GPU WARMUP
            if ii == 0:
                time.sleep(5)  # Allow dataloaders to cache a couple batches
                for _ in range(WARMUP_ITERS):
                    _ = model(local_batch)

            if args.only_global or args.precompute_descriptors:
                # Discard first measurement after cuDNN runs benchmarks
                anc_model_out = model(local_batch['anc_batch'], global_only=True)
                # MEASURE PERFORMANCE
                for rep in range(RUN_ITERS):
                    torch.cuda.synchronize()
                    # starter.record()
                    start = time.perf_counter()
                    _ = model(local_batch['anc_batch'], global_only=True)
                    # ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    # curr_time = starter.elapsed_time(ender)
                    curr_time = (end - start) * 1000  # convert to ms
                    global_timings.append(curr_time)

                if args.only_global:
                    continue
                
                # Pre-compute descriptors
                local_batch['anc_local_feats'] = anc_model_out['local']
                pos_model_out = model(local_batch['pos_batch'], global_only=True)
                local_batch['pos_local_feats'] = pos_model_out['local']

            # Discard first measurement after cuDNN runs benchmarks
            _ = model(local_batch)
            # MEASURE PERFORMANCE
            for rep in range(RUN_ITERS):
                torch.cuda.synchronize()
                # starter.record()
                start = time.perf_counter()
                _ = model(local_batch)
                # ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                end = time.perf_counter()
                # curr_time = starter.elapsed_time(ender)
                curr_time = (end - start) * 1000  # convert to ms
                metric_loc_timings.append(curr_time)

    # Re-ranking loop
    for ii, global_batch in tqdm.tqdm(enumerate(dataloaders['global_val']), total=len(dataloaders['global_val'])):
        if ii == args.max_samples:
            break
        batch = to_device(global_batch['batch'], device, non_blocking=True, construct_octree_neigh=True)
        shift_and_scale = to_device(global_batch['shift_and_scale'], device, non_blocking=True)
        with torch.inference_mode():
            # GPU WARMUP
            if ii == 0:
                time.sleep(5)  # Allow dataloaders to cache a couple batches
                for _ in range(WARMUP_ITERS):
                    _ = model(batch, global_only=True)

            # Get local features from backbone
            model_out = model(batch, global_only=True)
            
            # Discard first measurement after cuDNN runs benchmarks
            if params.model_params.rerank_mode == 'sgv':
                _ = model.sgv_rerank_inference(model_out, shift_and_scale, batch, feat_type='fine')
            else:
                _ = model.rerank_inference(model_out, shift_and_scale, batch)
                
            # MEASURE PERFORMANCE
            for rep in range(RUN_ITERS):
                torch.cuda.synchronize()
                # starter.record()
                start = time.perf_counter()
                if params.model_params.rerank_mode == 'sgv':
                    _ = model.sgv_rerank_inference(model_out, shift_and_scale, batch, feat_type='fine')
                else:
                    _ = model.rerank_inference(model_out, shift_and_scale, batch)
                # ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                end = time.perf_counter()
                # curr_time = starter.elapsed_time(ender)
                curr_time = (end - start) * 1000  # convert to ms
                rerank_timings.append(curr_time)

    return num_points, global_timings, metric_loc_timings, rerank_timings

def runtime_eval_egonn(params, device, model, dataloaders, n_kpts: int = 128):
    assert params.model_params.rerank_mode == 'sgv'

    # starter = torch.cuda.Event(enable_timing=True)
    # ender = torch.cuda.Event(enable_timing=True)
    global_timings = []
    metric_loc_timings = []
    rerank_timings = []
    num_points = []

    # PR + Metric Loc loop
    for ii, local_batch in tqdm.tqdm(enumerate(dataloaders['local_val']), total=len(dataloaders['local_val'])):
        if ii == args.max_samples:
            break
        num_points.append(local_batch['anc_batch']['pcd'][0].size(0))
        num_points.append(local_batch['pos_batch']['pcd'][0].size(0))
        local_batch = to_device(local_batch, device, non_blocking=True)
        with torch.inference_mode():
            # GPU WARMUP
            if ii == 0:
                time.sleep(5)  # Allow dataloaders to cache a couple batches
                for _ in range(WARMUP_ITERS):
                    _ = model(local_batch['anc_batch'])

            # Place Recognition
            # Discard first measurement after cuDNN runs benchmarks
            anc_model_out = model(local_batch['anc_batch'])
            # MEASURE PERFORMANCE
            for rep in range(RUN_ITERS):
                torch.cuda.synchronize()
                # starter.record()
                start = time.perf_counter()
                _ = model(local_batch['anc_batch'])
                # ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                end = time.perf_counter()
                # curr_time = starter.elapsed_time(ender)
                curr_time = (end - start) * 1000  # convert to ms
                global_timings.append(curr_time)

            if args.only_global:
                continue
                
            # Pre-compute descriptors
            pos_model_out = model(local_batch['pos_batch'])
            anc_descriptors, anc_keypoints, anc_sigma = (
                anc_model_out['descriptors'][0],
                anc_model_out['keypoints'][0],
                anc_model_out['sigma'][0],
            )
            pos_descriptors, pos_keypoints, pos_sigma = (
                pos_model_out['descriptors'][0],
                pos_model_out['keypoints'][0],
                pos_model_out['sigma'][0],
            )
            pos_n_kpts = min(len(pos_sigma), n_kpts)  # 128 keypoints by default
            _, pos_indices = torch.topk(pos_sigma.squeeze(1), dim=0, k=pos_n_kpts, largest=False)
            pos_descriptors_cpu = release_cuda(pos_descriptors[pos_indices])
            pos_keypoints_cpu = release_cuda(pos_keypoints[pos_indices])

            # Metric Localisation
            # MEASURE PERFORMANCE
            for rep in range(RUN_ITERS):
                torch.cuda.synchronize()
                # starter.record()
                start = time.perf_counter()

                # Get top-k keypoints (anchor will be done online, so it is within the timing loop)
                anc_n_kpts = min(len(anc_sigma), n_kpts)  # 128 keypoints by default
                _, anc_indices = torch.topk(anc_sigma.squeeze(1), dim=0, k=anc_n_kpts, largest=False)
                anc_descriptors_cpu = release_cuda(anc_descriptors[anc_indices])
                anc_keypoints_cpu = release_cuda(anc_keypoints[anc_indices])

                T_estimated, inliers, fitness = ransac_fn(
                    query_keypoints=anc_keypoints_cpu,
                    candidate_keypoints=pos_keypoints_cpu,
                    query_features=anc_descriptors_cpu,
                    candidate_features=pos_descriptors_cpu,
                    n_k=n_kpts,
                )
                
                # ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                end = time.perf_counter()
                # curr_time = starter.elapsed_time(ender)
                curr_time = (end - start) * 1000  # convert to ms
                metric_loc_timings.append(curr_time)

    # Re-ranking loop
    for ii, global_batch in tqdm.tqdm(enumerate(dataloaders['global_val']), total=len(dataloaders['global_val'])):
        if ii == args.max_samples:
            break
        batch = to_device(global_batch['batch'], device, non_blocking=True)
        shift_and_scale = to_device(global_batch['shift_and_scale'], device, non_blocking=True)
        with torch.inference_mode():
            # GPU WARMUP
            if ii == 0:
                time.sleep(5)  # Allow dataloaders to cache a couple batches
                for _ in range(WARMUP_ITERS):
                    _ = model(batch)

            # Get local features from backbone
            model_out = model(batch)
            descriptors, keypoints, sigma = (
                model_out['descriptors'],
                model_out['keypoints'],
                model_out['sigma'],
            )

            # Sort by saliency
            for ii in range(len(descriptors)):
                n_kpts_tmp = min(len(sigma[ii]), n_kpts)  # 128 keypoints by default
                _, indices = torch.topk(sigma[ii].squeeze(1), dim=0, k=n_kpts_tmp, largest=False)
                descriptors[ii] = descriptors[ii][indices]
                keypoints[ii] = keypoints[ii][indices]
            
            # Re-ranking
            # MEASURE PERFORMANCE
            for rep in range(RUN_ITERS):
                torch.cuda.synchronize()
                # starter.record()
                start = time.perf_counter()

                query_keypoints = keypoints[0][None, ...]
                candidate_keypoints = torch.stack(keypoints[1:], dim=0)
                query_features = descriptors[0][None, ...]
                candidate_features = torch.stack(descriptors[1:], dim=0)
                rerank_scores = torch.tensor(sgv_parallel(
                    src_keypts=query_keypoints,
                    tgt_keypts=candidate_keypoints,
                    src_features=query_features,
                    tgt_features=candidate_features,
                    d_thresh=params.sgv_d_thresh,
                ))
                
                # ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                end = time.perf_counter()
                # curr_time = starter.elapsed_time(ender)
                curr_time = (end - start) * 1000  # convert to ms
                rerank_timings.append(curr_time)

    return num_points, global_timings, metric_loc_timings, rerank_timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise correspondences from HOTFormerMetricLoc')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--only_global', action='store_true', help='Only run global descriptor forward pass (only on one submap instead of a pair)')
    parser.add_argument('--precompute_descriptors', action='store_true', help='Pre-computes global descriptors to simulate online inference')
    parser.add_argument('--num_neighbours', type=int, default=20, help='Num neighbours to consider for re-ranking')
    parser.add_argument('--use_sgv', action='store_true', help='Use SGV for re-ranking')
    parser.add_argument('--max_samples', type=int, default=-1, help='Max number of samples from validation set to time (defaults to all).')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Only global: {}'.format(args.only_global))
    print('Pre-compute descriptors: {}'.format(args.precompute_descriptors))
    print('Num neighbours: {}'.format(args.num_neighbours))
    print(f'Use SGV: {args.use_sgv}')
    if not (args.max_samples == -1 or args.max_samples > 0):
        raise ValueError
    print('Max samples: {}'.format(args.max_samples))
    print('Debug mode: {}'.format(args.debug))
    print('')

    set_seed()  # Seed RNG
    # Enable benchmark mode
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    main()