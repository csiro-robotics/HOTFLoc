"""
Runtime evaluation for HOTFormerMetricLoc on real data (validation set).
"""
import argparse
import os
import time

import tqdm
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from dataset.dataset_utils import make_dataloaders
from models.model_factory import model_factory
from misc.torch_utils import set_seed, release_cuda, to_device
from misc.utils import TrainingParams

WARMUP_ITERS = 10
RUN_ITERS = 3


def setup_model():
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
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
    model.benchmark()

    # Get val dataloader with batch size 1
    params.val_batch_size = 1
    params.local.batch_size = 1
    params.num_workers = 2
    dataloaders = make_dataloaders(params, local=True, validation=True)
    dataloader = dataloaders['local_val']

    return params, device, model, dataloader

def main():
    params, device, model, dataloader = setup_model()

    # starter = torch.cuda.Event(enable_timing=True)
    # ender = torch.cuda.Event(enable_timing=True)
    global_timings = []
    metric_loc_timings = []
    num_points = []

    for ii, local_batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if ii == args.max_samples:
            break
        num_points.append(local_batch['anc_batch']['points'].npt)
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
                
                # Pre-compute global descriptors
                local_batch['anc_local_feats'] = {
                    'coarse': anc_model_out['local'][model.depth_coarse],
                    'fine': anc_model_out['local'][model.depth_fine],
                }
                pos_model_out = model(local_batch['pos_batch'], global_only=True)
                local_batch['pos_local_feats'] = {
                    'coarse': pos_model_out['local'][model.depth_coarse],
                    'fine': pos_model_out['local'][model.depth_fine],
                }

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

    num_points = np.asarray(num_points)
    print(f"Pointcloud size: mean - {num_points.mean():.0f} pts, std - {num_points.std():.0f} pts")
    if len(global_timings) > 0:
        global_timings = np.asarray(global_timings)
        print(f"RUNTIME (global descriptor): mean - {global_timings.mean():.2f}ms, std - {global_timings.std():.2f}ms")
    metric_loc_timings = np.asarray(metric_loc_timings)
    print(f"RUNTIME (metric loc): mean - {metric_loc_timings.mean():.2f}ms, std - {metric_loc_timings.std():.2f}ms")

    print(f"Max mem allocated {torch.cuda.max_memory_allocated(device=None) / (1024 ** 2):.2f} MB memory")
    # print(f"mem allocated {torch.cuda.memory_allocated(device=None) / (1024 ** 2):.2f} MB memory")
    torch.cuda.reset_peak_memory_stats(device=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise correspondences from HOTFormerMetricLoc')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--only_global', action='store_true', help='Only run global descriptor forward pass (only on one submap instead of a pair)')
    parser.add_argument('--precompute_descriptors', action='store_true', help='Pre-computes global descriptors to simulate online inference')
    parser.add_argument('--max_samples', type=int, default=-1, help='Max number of samples from validation set to time (defaults to all).')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Only global: {}'.format(args.only_global))
    print('Pre-compute descriptors: {}'.format(args.precompute_descriptors))
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