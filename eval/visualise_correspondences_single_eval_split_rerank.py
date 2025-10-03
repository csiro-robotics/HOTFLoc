"""
Visualises correspondences from HOTFormerMetricLoc on a single evaluation split.
"""
import argparse
import pickle
import os

import numpy as np
import torch
import tqdm
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt

from dataset.dataset_utils import make_eval_dataloader_6DOF, make_eval_dataloader_reranking
from dataset.augmentation import Normalize
from eval.vis_utils import (
    colourise_points_by_similarity,
    visualise_coarse_correspondences,
    visualise_fine_correspondences,
    visualise_points,
    visualise_registration,
    visualise_LGR_initial_registration,
    visualise_similarity,
)
from eval.evaluate_metric_loc_splits_rerank import load_embeddings_from_file, save_embeddings_to_file, get_latent_vectors
from eval.sgv.sgv_utils import sgv_parallel
from eval.utils import get_query_database_splits
from models.egonn import MinkGL as EgoNN
from models.model_factory import model_factory
from models.losses.geotransformer_loss import Evaluator
from misc.torch_utils import set_seed, release_cuda, to_device
from misc.poses import gravity_align_pc_with_pose
from misc.utils import TrainingParams
from misc.logger import create_logger
from geotransformer.utils.visualization import (
    draw_point_to_node,
    draw_node_correspondences,
    get_colors_with_tsne
)

EVAL_MODES = ['Initial', 'Re-Ranked']


def get_eval_pairs_dataloader(
    params: TrainingParams,
    model: torch.nn.Module,
    model_name: str,
    device: str,
    reranking: bool,
):
    # Load query and db pickles
    # Extract location name from query and database files
    eval_database_files, eval_query_files = get_query_database_splits(params)
    assert len(eval_database_files) == len(eval_query_files)
    database_file = eval_database_files[args.dataset_idx]
    query_file = eval_query_files[args.dataset_idx]
    if 'AboveUnder' in params.dataset_name or 'CSWildPlaces' in params.dataset_name:
        if "pickles/" in database_file:  # CS-WildPlaces
            location_name = database_file.split('/')[-1].split('_')[1]
            temp = query_file.split('/')[-1].split('_')[1]
        else:
            location_name = database_file.split('_')[1]
            temp = query_file.split('_')[1]
    else:
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
    assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                    query_file)
    print(f'Location name: {location_name}')
    if args.save_dir is not None:
        # Add location name to save path
        args.save_dir = os.path.join(args.save_dir, location_name)
        os.makedirs(args.save_dir, exist_ok=True)

    p = os.path.join(params.dataset_folder, database_file)
    with open(p, 'rb') as f:
        database_sets = pickle.load(f)

    p = os.path.join(params.dataset_folder, query_file)
    with open(p, 'rb') as f:
        query_sets = pickle.load(f)

    if (args.database_split_idx == args.query_split_idx) and params.skip_same_run:
        raise ValueError('Invalid query/db combo for given dataset')
    database_set = database_sets[args.database_split_idx]
    query_set = query_sets[args.query_split_idx]
    
    if params.dataset_name == 'WildPlaces':
        # For WildPlaces, there are multiple databases per query set, so add both to split name
        split_name = (os.path.split(os.path.split(database_set[0]['query'])[0])[0]
                      + '-' + query_set[0]['query'].split('/')[1])
    else:
        split_name = os.path.split(os.path.split(query_set[0]['query'])[0])[0]
    print(f'Split name: {split_name}')

    # Load/process database global embeddings
    print(f'{"Loading" if args.load_embeddings else "Computing"} database embeddings')
    database_positions = np.array([(db_details['northing'], db_details['easting']) for db_details in database_set.values()])
    if args.load_embeddings:
        database_embeddings, database_local_dict = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'database_{args.database_split_idx}')
    else:
        database_embeddings, database_local_dict = get_latent_vectors(model, database_set, device, params, only_global=False, show_progress=True)
    if args.save_embeddings:
        save_embeddings_to_file(database_embeddings, database_local_dict, model_name, params.dataset_name, location_name, f'database_{args.database_split_idx}')

    if args.load_embeddings:
        query_embeddings, query_local_dict = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'query_{args.query_split_idx}')
    else:
        query_embeddings, query_local_dict = get_latent_vectors(model, query_set, device, params, only_global=False, show_progress=True)
    if args.save_embeddings:
        save_embeddings_to_file(query_embeddings, query_local_dict, model_name, params.dataset_name, location_name, f'query_{args.query_split_idx}')

    # Compute PR results to determine metric loc pairs
    global_metrics = {}
    num_neighbors = 20
    radius = params.eval_radius
    intermediate_metrics = {
        'tp': {r: [0] * num_neighbors for r in radius},
        'tp_rr': {r: [0] * num_neighbors for r in radius},
        'opr': {r: 0 for r in radius},
        'opr_rr': {r: 0 for r in radius},
        'RR': {r: [] for r in radius},
        'RR_rr': {r: [] for r in radius},
    }

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_global_nbrs = KDTree(database_embeddings)

    metric_loc_pairs_list = {eval_mode: [] for eval_mode in EVAL_MODES}
    rerank_eigvec_dict = {}

    opr_threshold = max(int(round(len(database_embeddings)/100.0)), 1)

    num_evaluated = 0
    global_result_dict = {
        'query_nn_list': [],
        'euclid_dist_list': [],
    }
    for query_idx in tqdm.tqdm(range(len(query_embeddings)),
                               desc='Place Recognition',):
        if len(args.idx_list) > 0 and query_idx not in args.idx_list:
            continue
        query_metadata = query_set[query_idx]  # {'query': path, 'northing': , 'easting': , 'pose': }
        query_position = np.array((query_metadata['northing'], query_metadata['easting']))
        if args.database_split_idx in query_metadata:  # old tuples store true neighbours directly
            true_neighbors = query_metadata[args.database_split_idx]
            if len(true_neighbors) == 0:
                continue
        else:  # expected that new tuples filter queries, but do here just in case
            min_neighbor_dist = np.linalg.norm(query_position - database_positions, axis=-1).min()
            if min_neighbor_dist > min(radius):
                continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, nn_indices = database_global_nbrs.query(  # (1, k) arrays
            query_embeddings[query_idx][None, :], k=num_neighbors,
        )
        distances, nn_indices = distances[0], nn_indices[0]  # (k,) arrays

        # Euclidean distance between the query and nn
        delta = query_position - database_positions[nn_indices]  # (k, 2) array
        euclid_dist = np.linalg.norm(delta, axis=-1)  # (k,) array

        # Save candidates for re-ranking
        if reranking:
            assert num_neighbors >= params.rerank_num_neighbours, 'Set num_neighbours higher'
            global_result_dict['query_nn_list'].append((query_idx, nn_indices[:params.rerank_num_neighbours]))
            global_result_dict['euclid_dist_list'].append(euclid_dist)

        # Count true positives and 1% retrieved for different radius and NN number
        intermediate_metrics['tp'] = {r: [intermediate_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in radius}
        intermediate_metrics['opr'] = {r: intermediate_metrics['opr'][r] + (1 if (euclid_dist[:opr_threshold] <= r).any() else 0) for r in radius}
        intermediate_metrics['RR'] = {r: intermediate_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in radius}

        if reranking:
            continue

        # LOCAL DESCRIPTOR EVALUATION
        # Do the evaluation only if the nn pose is within distance threshold
        # Otherwise the overlap is too small to get reasonable results
        # (evaluation continues if standard OR re-ranked nn pose is within thresh meters)
        if euclid_dist[0] > params.local.max_eval_threshold:  # and euclid_dist_rr[0] > local_max_eval_threshold:
            continue

        # Cache query and nn idx for metric loc eval
        metric_loc_pairs_list['Initial'].append((query_idx, nn_indices[0]))

    # Run re-ranking evaluation using pre-computed nearest neighbours
    # NOTE: Could be faster if done in the PR loop, but requires additional implementation
    #       to pre-compute local point coords (or octrees). Usable for now.
    if reranking and params.model_params.rerank_mode is not None:
        if isinstance(model, EgoNN):  # Don't need full dataloader for EgoNN
            rerank_dataloader = range(len(global_result_dict['query_nn_list']))
        else:
            rerank_dataloader = make_eval_dataloader_reranking(  # Increased num workers to minimise bottleneck (assumes enough threads are available)
                params, query_set, database_set, global_result_dict['query_nn_list'], num_workers=(params.num_workers * 3)
            )
        for idx, rerank_batch_dict in tqdm.tqdm(enumerate(rerank_dataloader),
                                                total=len(rerank_dataloader),
                                                desc='Re-ranking'):
            query_idx = global_result_dict['query_nn_list'][idx][0]
            nn_indices = global_result_dict['query_nn_list'][idx][1]
            euclid_dist = global_result_dict['euclid_dist_list'][idx]
            query_metadata = query_set[query_idx]  # {'query': path, 'northing': , 'easting': , 'pose': }
            query_position = np.array((query_metadata['northing'], query_metadata['easting']))

            # Separate forward pass for EgoNN+SGV
            eigvec_list = None
            if isinstance(model, EgoNN):
                assert params.model_params.rerank_mode == 'sgv'
                query_keypoints = query_local_dict['keypoints'][query_idx][None, ...]
                candidate_keypoints = torch.stack(
                    [database_local_dict['keypoints'][nn_idx] for nn_idx in nn_indices],
                    dim=0,
                )
                query_features = query_local_dict['local_embeddings'][query_idx][None, ...]
                candidate_features = torch.stack(
                    [database_local_dict['local_embeddings'][nn_idx] for nn_idx in nn_indices],
                    dim=0,
                )
                query_keypoints, candidate_keypoints, query_features, candidate_features = to_device(
                    (query_keypoints, candidate_keypoints, query_features, candidate_features),
                    device=device,
                )
                rerank_scores = torch.tensor(sgv_parallel(
                    src_keypts=query_keypoints,
                    tgt_keypts=candidate_keypoints,
                    src_features=query_features,
                    tgt_features=candidate_features,
                    d_thresh=0.4,  # threshold used in SGV paper
                ))
            # HOTFormerLoc-based re-ranking
            else:
                # Move to GPU and do forward pass
                rerank_batch_dict = to_device(rerank_batch_dict, device, non_blocking=True, construct_octree_neigh=True)
                rerank_batch = rerank_batch_dict['batch']
                rerank_shift_and_scale = rerank_batch_dict['shift_and_scale']
                with torch.inference_mode():
                    if params.model_params.rerank_mode in ('relay_token_gc', 'relay_token_local_gc'):
                        out_dict = model(rerank_batch, global_only=True)
                    elif params.model_params.rerank_mode in ('local_hierarchical_gc', 'sgv'):
                        sgv_feat_type = 'fine'  # TODO: make this a configurable param
                        # Use pre-computed local descriptors
                        rerank_batch_local_embeddings = [
                            query_local_dict['local_embeddings'][query_idx],
                            *[database_local_dict['local_embeddings'][nn_idx] for nn_idx in nn_indices]
                        ]
                        out_dict = {'local': rerank_batch_local_embeddings}
                        out_dict = to_device(out_dict, device)
                    if params.model_params.rerank_mode == 'sgv':
                        rerank_dict = model.sgv_rerank_inference(
                            model_out=out_dict,
                            shift_and_scale=rerank_shift_and_scale,
                            batch=rerank_batch,
                            feat_type=sgv_feat_type,
                        )
                        rerank_scores = rerank_dict['scores']
                        eigvec_list = rerank_dict['eigvec_list']
                    else:
                        rerank_dict = model.rerank_inference(
                            model_out=out_dict,
                            shift_and_scale=rerank_shift_and_scale,
                            batch=rerank_batch,
                        )
                        rerank_scores = rerank_dict['scores'][0, :, 0]
                        eigvec_list = rerank_dict['eigvec_list'][0]

            _, rerank_sort_indices = release_cuda(
                torch.sort(rerank_scores, descending=True), to_numpy=True
            )
            topk_rerank_indices = nn_indices[rerank_sort_indices]
            delta_rerank = query_position - database_positions[topk_rerank_indices]
            euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)

            intermediate_metrics['tp_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in radius}
            intermediate_metrics['opr_rr'] = {r: intermediate_metrics['opr_rr'][r] + (1 if (euclid_dist_rr[:opr_threshold] <= r).any() else 0) for r in radius}
            intermediate_metrics['RR_rr'] = {r: intermediate_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in radius}

            # LOCAL DESCRIPTOR EVALUATION
            # Do the evaluation only if the nn pose is within distance threshold
            # Otherwise the overlap is too small to get reasonable results
            # (evaluation continues if standard OR re-ranked nn pose is within thresh meters)
            if (euclid_dist[0] > params.local.max_eval_threshold
                and euclid_dist_rr[0] > params.local.max_eval_threshold):
                continue

            # Cache query and nn idx for metric loc eval
            metric_loc_pairs_list['Initial'].append((query_idx, nn_indices[0]))
            metric_loc_pairs_list['Re-Ranked'].append((query_idx, topk_rerank_indices[0]))

            # Also store eigvec for later
            if eigvec_list is not None:
                rerank_eigvec_dict[query_idx] = {
                    'Initial': release_cuda(eigvec_list[nn_indices[0]]),
                    'Re-Ranked': release_cuda(eigvec_list[rerank_sort_indices[0]])
                }

    # Calculate mean global metrics
    global_metrics['recall'] = {r: [intermediate_metrics['tp'][r][nn] / num_evaluated for nn in range(num_neighbors)] for r in radius}
    global_metrics['recall@1'] = {r: global_metrics['recall'][r][0] for r in radius}
    global_metrics['recall@1%'] = {r: intermediate_metrics['opr'][r] / num_evaluated for r in radius}
    global_metrics['MRR'] = {r: np.mean(np.asarray(intermediate_metrics['RR'][r])) for r in radius}
    if reranking:
        global_metrics['recall_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] / num_evaluated for nn in range(num_neighbors)] for r in radius}
        global_metrics['recall@1_rr'] = {r: global_metrics['recall_rr'][r][0] for r in radius}
        global_metrics['recall@1%_rr'] = {r: intermediate_metrics['opr_rr'][r] / num_evaluated for r in radius}
        global_metrics['MRR_rr'] = {r: np.mean(np.asarray(intermediate_metrics['RR_rr'][r])) for r in radius}

    msg = 'Place Recognition Results:'
    msg += '\nINITIAL'
    recall = global_metrics['recall']
    for radius in recall.keys():
        msg += f"\n  Radius: {radius} [m]:\n    Recall@N: "
        for ii, x in enumerate(recall[radius]):
                msg += f"{x:0.3f}, "
                if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                    msg += "\n              "
        msg += "\n    Recall@1%: {:0.3f}".format(global_metrics['recall@1%'][radius])
        msg += '\n    MRR: {:0.3f}'.format(global_metrics['MRR'][radius])
    if reranking:
        msg += '\nRE-RANKED'
        recall_rr = global_metrics['recall_rr']
        for radius in recall_rr.keys():
            msg += f"\n  Radius: {radius} [m]:\n    Recall@N: "
            for ii, x in enumerate(recall_rr[radius]):
                    msg += f"{x:0.3f}, "
                    if (ii+1) % 5 == 0 and (ii+1) < len(recall_rr[radius]):
                        msg += "\n              "
            msg += "\n    Recall@1%: {:0.3f}".format(global_metrics['recall@1%_rr'][radius])
            msg += '\n    MRR: {:0.3f}'.format(global_metrics['MRR_rr'][radius])
    print(msg)

    eval_mode = 'Re-Ranked' if reranking else 'Initial'
    eval_dataloader = make_eval_dataloader_6DOF(
        params, query_set, database_set, metric_loc_pairs_list[eval_mode]
    )
    return eval_dataloader, query_local_dict, database_local_dict, rerank_eigvec_dict

def setup_model():
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
    model_name = os.path.splitext(os.path.split(args.weights)[1])[0]
    print('Loading weights: {}'.format(args.weights))
    try:
        if os.path.splitext(args.weights)[1] == '.ckpt':
            state = torch.load(args.weights)
            model.load_state_dict(state['model_state_dict'])
        else:  # .pt or .pth
            model.load_state_dict(torch.load(args.weights, map_location=device))
    except RuntimeError:
        # Try legacy mode
        model = model_factory(params, legacy=True)
        if os.path.splitext(args.weights)[1] == '.ckpt':
            state = torch.load(args.weights)
            model.load_state_dict(state['model_state_dict'])
        else:  # .pt or .pth
            model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)
    model.eval()

    logging_level = 'DEBUG' if params.debug else 'INFO'
    create_logger(log_file=None, logging_level=logging_level)

    # Check if re-ranking
    reranking = False
    if not args.disable_reranking:
        if params.model_params.rerank_mode is not None or args.use_sgv:
            reranking = True
            if args.use_sgv:
                params.model_params.rerank_mode = 'sgv'

    params.local.eval_num_workers = args.num_workers
    (
        dataloader,
        query_local_dict,
        database_local_dict,
        rerank_eigvec_dict
    ) = get_eval_pairs_dataloader(params, model, model_name, device, reranking)
    # dataloader.shuffle = True  # NOTE: setting to true will invalidate metric_loc_pairs_list order
    
    return params, device, model, dataloader, query_local_dict, database_local_dict, rerank_eigvec_dict, reranking

def main():
    (
        params,
        device,
        model,
        dataloader,
        query_local_dict,
        database_local_dict,
        rerank_eigvec_dict,
        reranking
    ) = setup_model()

    # Get evaluator
    metric_loc_evaluator = Evaluator(params)
    metric_loc_pairs_list = dataloader.dataset.pairs_list

    # TODO: Fix rest of code to viz what we want
    #       - Top-down view of query vs retrieved candidate (or whatever is most visible)
    #       - Probably coarse corrs and fine corrs for good measure? (but only use global inlier fine corrs)

    for ii, local_batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        query_idx, nn_idx = metric_loc_pairs_list[ii]
        if len(args.idx_list) > 0 and query_idx not in args.idx_list:
            continue
        local_batch['anc_local_feats'] = query_local_dict['local_embeddings'][query_idx]
        local_batch['pos_local_feats'] = database_local_dict['local_embeddings'][nn_idx]
        local_batch = to_device(local_batch, device, non_blocking=True, construct_octree_neigh=True)
        with torch.inference_mode():
            output_dict = model(local_batch)[0]

        batch_temp = {'transform': local_batch['transform'][0]}  # temp fix since loss func expects a single batch item
        eval_metrics = release_cuda(metric_loc_evaluator(output_dict, batch_temp))

        # Skip successes or failures
        success = eval_metrics['RR'] > 0
        if (success and args.failures) or (not success and args.successes):
            continue

        # Undo cylindrical projection and normalisation
        anc_points_raw = release_cuda(local_batch['anc_batch']['points'].points)
        pos_points_raw = release_cuda(local_batch['pos_batch']['points'].points)
        if model.quantizer is not None:
            anc_points_raw = model.quantizer.undo_conversion(anc_points_raw)
            pos_points_raw = model.quantizer.undo_conversion(pos_points_raw)
        anc_points_raw = Normalize.unnormalize(
            anc_points_raw, release_cuda(local_batch['anc_shift_and_scale'][0])
        )
        pos_points_raw = Normalize.unnormalize(
            pos_points_raw, release_cuda(local_batch['pos_shift_and_scale'][0])
        )
        anc_points_coarse = release_cuda(output_dict['anc_points_coarse'])
        anc_points_fine = release_cuda(output_dict['anc_points_fine'])
        anc_point_to_node = release_cuda(output_dict['anc_point_to_node'])
        anc_feats_coarse = release_cuda(output_dict['anc_feats_coarse'])
        anc_feats_coarse_pre_refinement = release_cuda(output_dict['anc_feats_coarse_pre_refinement'])
        anc_feats_fine = release_cuda(output_dict['anc_feats_fine'])
        pos_points_coarse = release_cuda(output_dict['pos_points_coarse'])
        pos_points_fine = release_cuda(output_dict['pos_points_fine'])
        pos_point_to_node = release_cuda(output_dict['pos_point_to_node'])
        pos_feats_coarse = release_cuda(output_dict['pos_feats_coarse'])
        pos_feats_coarse_pre_refinement = release_cuda(output_dict['pos_feats_coarse_pre_refinement'])
        pos_feats_fine = release_cuda(output_dict['pos_feats_fine'])
        gt_node_corr_indices = release_cuda(output_dict['gt_node_corr_indices'])
        gt_node_corr_overlaps = release_cuda(output_dict['gt_node_corr_overlaps'])  # NOTE: Only averaged on non-zero overlap patches
        anc_node_corr_indices = release_cuda(output_dict['anc_node_corr_indices'])
        pos_node_corr_indices = release_cuda(output_dict['pos_node_corr_indices'])
        node_corr_indices = torch.stack((anc_node_corr_indices, pos_node_corr_indices), dim=1)
        anc_node_corr_knn_masks = release_cuda(output_dict['anc_node_corr_knn_masks'])
        pos_node_corr_knn_masks = release_cuda(output_dict['pos_node_corr_knn_masks'])
        anc_corr_points = release_cuda(output_dict['anc_corr_points'])
        pos_corr_points = release_cuda(output_dict['pos_corr_points'])
        best_anc_corr_points = release_cuda(output_dict['best_anc_corr_points'])
        best_anc_corr_points = best_anc_corr_points[torch.nonzero(best_anc_corr_points)[:,0].unique()]
        best_pos_corr_points = release_cuda(output_dict['best_pos_corr_points'])
        best_pos_corr_points = best_pos_corr_points[torch.nonzero(best_pos_corr_points)[:,0].unique()]
        best_corr_scores = release_cuda(output_dict['best_corr_scores'])
        best_corr_scores = best_corr_scores[torch.nonzero(best_corr_scores)[:,0].unique()].numpy()
        T_best_corr = release_cuda(output_dict['best_corr_transform'])
        matching_scores = release_cuda(output_dict['matching_scores'], to_numpy=True)
        corr_scores = release_cuda(output_dict['corr_scores'], to_numpy=True)
        num_corr_points = np.array(output_dict['num_corr_points'])
        T_gt = release_cuda(local_batch['transform'][0])
        T_estimated = release_cuda(output_dict['estimated_transform'])

        num_points_per_patch = torch.concat(
            (anc_node_corr_knn_masks.float().sum(1),
             pos_node_corr_knn_masks.float().sum(1),),
            dim=0,
        )
        # Only measure horizontal distance
        pos_dist = T_gt[:2,3].norm().item()
        
        # Print metrics as well
        log_str = f"Query ID: {query_idx} -- nn ID: {nn_idx}"
        log_str += f' -- nn dist: {pos_dist:.2f}m'
        log_str += f' -- RTE: {eval_metrics['RTE']:.2f}m -- RRE: {eval_metrics['RRE']:.2f}deg'
        log_str += f' -- Coarse IR: {eval_metrics['PIR']*100:.1f}% -- Fine IR: {eval_metrics['IR']*100:.1f}%'
        log_str += f' -- mean pts per patch: {num_points_per_patch.mean():.1f} (±{num_points_per_patch.std():.1f})'
        log_str += f' -- mean patch overlap: {gt_node_corr_overlaps.mean()*100:.1f}% (±{gt_node_corr_overlaps.std()*100:.1f}%)'
        log_str += f' -- num corr patches (after OT): {len(num_corr_points)}'
        log_str += f' -- mean corr pts per patch: {num_corr_points.mean():.1f} (±{num_corr_points.std():.1f})'
        log_str += f' -- mean corr score: {corr_scores.mean():.3f} (±{corr_scores.std():.3f})'
        if args.verbose:
            log_str += f'\n  query file: {dataloader.dataset.data_set_dict[query_idx]['query']}'
            log_str += f' -- nn file: {dataloader.dataset.pos_dataset.data_set_dict[nn_idx]['query']}'
            log_str += f'\n  GT TF:\n{T_gt.numpy()}'
            log_str += f'\n  Est TF:\n{T_estimated.numpy()}'
        print(log_str, flush=True)

        save_dir_ii = None
        if args.save_dir is not None:
            save_dir_ii = os.path.join(args.save_dir, f'{query_idx}')
            save_dir_ii += f'-reranked-{params.model_params.rerank_mode}' if reranking else '-initial'
            save_dir_ii += f'-dist{pos_dist:.1f}m'
            save_dir_ii += '-success' if success else '-fail'
            os.makedirs(save_dir_ii, exist_ok=True)
            # Save stats to dir for future reference:
            with open(os.path.join(save_dir_ii, 'stats.txt'), 'w') as f:
                f.write(log_str)

        if not args.registration_only:
            visualise_coarse_correspondences(
                anc_points_coarse=anc_points_coarse,
                pos_points_coarse=pos_points_coarse,
                # anc_points_fine=anc_points_fine,
                # pos_points_fine=pos_points_fine,
                anc_points_fine=anc_points_raw,
                pos_points_fine=pos_points_raw,
                node_corr_indices=node_corr_indices.numpy(),
                gt_node_corr_indices=gt_node_corr_indices.numpy(),
                transform=T_gt.numpy(),
                # anc_point_to_node=anc_point_to_node,
                # pos_point_to_node=pos_point_to_node,
                anc_feats_coarse=anc_feats_coarse,
                pos_feats_coarse=pos_feats_coarse,
                translate=[0,0,50],
                zoom=args.zoom,
                plot_coarse=True,  # Plot keypoints as spheres
                show_unused=False,
                coarse_colourmode='height',
                # coarse_colourmode='patch',
                # coarse_colourmode='tsne',
                # coarse_colourmode='umap',
                save_dir=save_dir_ii,
                disable_animation=args.disable_animation,
                non_interactive=args.non_interactive,
                voxel_size=args.voxel_size,
            )

            visualise_fine_correspondences(
                anc_points_fine=anc_points_fine,
                pos_points_fine=pos_points_fine,
                anc_corr_points=anc_corr_points,
                pos_corr_points=pos_corr_points,
                corr_scores=corr_scores,
                transform=T_gt.numpy(),
                score_threshold=0.15,
                anc_feats_fine=anc_feats_fine,
                pos_feats_fine=pos_feats_fine,
                translate=[0,0,50],
                zoom=args.zoom,
                colourmode='height',
                # colourmode='umap',
                save_dir=save_dir_ii,
                disable_animation=args.disable_animation,
                non_interactive=args.non_interactive,
            )

        # Plot point clouds on their own (gravity align query to correct top-down view on CSWP)
        anc_points_raw_grav_aligned, _ = gravity_align_pc_with_pose(anc_points_raw, T_gt.numpy())
        visualise_points(
            points=anc_points_raw_grav_aligned,
            # transform=T_gt.numpy(),  # use GT TF to remove rotation offset
            transform=np.eye(4),
            zoom=(args.zoom - 0.10),
            angle=0, # top-down view
            save_dir=save_dir_ii,
            filename='query_pc',
            disable_animation=True,
            non_interactive=args.non_interactive,
            voxel_size=None,
        )
        visualise_points(
            points=pos_points_raw,
            transform=np.eye(4),
            zoom=(args.zoom - 0.10),
            angle=0, # top-down view
            save_dir=save_dir_ii,
            filename='nn_pc',
            disable_animation=True,
            non_interactive=args.non_interactive,
            voxel_size=None,
        )
        # Plot eigenvectors from re-ranking
        if reranking:
            eigvec = rerank_eigvec_dict[query_idx]['Re-Ranked']
            fig, ax = plt.subplots(figsize=(2,5))
            ax.imshow(eigvec[:, None].expand(-1, len(eigvec)//10), cmap='RdYlGn')
            ax.set_xticks([])
            fig.savefig(os.path.join(save_dir_ii, 'eigvec.png'))

        # Ground truth TF (with ICP, if enabled in params)
        visualise_registration(
            anc_points_fine=anc_points_raw,
            pos_points_fine=pos_points_raw,
            transform=T_gt.numpy(),
            zoom=(args.zoom - 0.10),
            save_dir=save_dir_ii,
            filename='registration_GT',
            disable_animation=True,
            non_interactive=args.non_interactive,
            voxel_size=args.voxel_size,
        )

        # Estimated TF
        visualise_registration(
            anc_points_fine=anc_points_raw,
            pos_points_fine=pos_points_raw,
            transform=T_estimated.numpy(),
            zoom=(args.zoom - 0.10),
            save_dir=save_dir_ii,
            filename='registration',
            disable_animation=args.disable_animation,
            non_interactive=args.non_interactive,
            voxel_size=args.voxel_size,
        )

        pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise correspondences from HOTFormerMetricLoc')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--dataset_idx', type=int, required=True, help='Index of desired dataset (as per eval/utils.py)')
    parser.add_argument('--database_split_idx', type=int, required=True, help='Index of desired database split')
    parser.add_argument('--query_split_idx', type=int, required=True, help='Index of desired query split')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--save_dir', type=str, required=False, help='Save visualisations/pcds to directory (creates subdirectories for each model)')
    parser.add_argument('--failures', action='store_true', help='Only visualise metric loc failures')
    parser.add_argument('--successes', action='store_true', help='Only visualise metric loc successes')
    parser.add_argument('--registration_only', action='store_true', help='Only visualise registration')
    parser.add_argument('--idx_list', type=int, nargs='+', default=[], help='Only visualise given list of indices (ordered by val dataloader)')
    parser.add_argument('--zoom', type=float, default=0.55, help='Zoom level for open3d viewer')
    parser.add_argument('--voxel_size', type=float, required=True, help='Voxel size to use for registration viz')
    parser.add_argument('--disable_animation', action='store_true', help='Disables animations')
    parser.add_argument('--non_interactive', action='store_true', help='Saves visualisations instantly, then closes viewer.')
    parser.add_argument('--use_sgv', action='store_true', help='Use SGV for re-ranking')
    parser.add_argument('--disable_reranking', action='store_true', help='Disable re-ranking evaluation to save time')
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings to disk')
    parser.add_argument('--load_embeddings', action='store_true',
                        help=('Load embeddings from disk. Note this script will only check if '
                              'weights paths match, not if the configs used match.'))
    parser.add_argument('--num_workers', type=int, default=2, help='Override num workers for eval dataloader')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print(f'Dataset split idx: {args.dataset_idx}')
    print(f'Database split idx: {args.database_split_idx}')
    print(f'Query split idx: {args.query_split_idx}')
    print(f'Use SGV: {args.use_sgv}')
    print(f'Disable re-ranking: {args.disable_reranking}')
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    if args.save_dir is not None:
        if args.weights is not None and 'weights/' in args.weights:
            model_pathname = (
                f"{os.path.splitext(args.weights.split('weights/')[-1])[0]}"
                f"--{os.path.splitext(os.path.split(args.config)[-1])[0]}"
            )
            args.save_dir = os.path.join(args.save_dir, model_pathname)
            os.makedirs(args.save_dir, exist_ok=True)
        else:
            raise ValueError('Must provide valid path to weights if `--save_dir` is set')
    print('Save dir: {}'.format(args.save_dir))
    print('Failures only: {}'.format(args.failures))
    print('Successes only: {}'.format(args.successes))
    assert not (args.failures and args.successes), "Pick failures OR successes"
    print('Registration only: {}'.format(args.registration_only))
    print('Idx list: {}'.format(args.idx_list))
    if args.failures and len(args.idx_list) > 0:
        print('WARNING: Ignoring `--failures` as `--idx_list` was specified')
        args.failures = False
    print('Zoom {}'.format(args.zoom))
    print('Voxel size {}'.format(args.voxel_size))
    print('Disable animation: {}'.format(args.disable_animation))
    print('Non interactive: {}'.format(args.non_interactive))
    if args.save_embeddings and args.load_embeddings:
        print('[WARNING] Both save_embeddings AND load_embeddings specified, which is redundant. '
              'Will proceed without saving embeddings.')
        args.save_embeddings = False
    print(f'Save embeddings: {args.save_embeddings}')
    print(f'Load embeddings: {args.load_embeddings}')
    print(f'Num workers: {args.num_workers}')
    assert args.num_workers >= 0, "Invalid num_workers"
    print('Debug mode: {}'.format(args.debug))
    print('Verbose mode: {}'.format(args.verbose))
    print('')

    set_seed()  # Seed RNG
    main()