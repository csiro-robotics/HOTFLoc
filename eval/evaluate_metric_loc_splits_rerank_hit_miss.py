# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
# Adapted to process samples in batches by Ethan Griffiths (QUT & CSIRO Data61).
# 24-05-2025 Further adapted to implement SpectralGV re-ranking: https://github.com/csiro-robotics/SpectralGV/blob/main
# 29-07-2025 Adapted to include metric localisation evaluation
# 03-09-2025 Adapted to include relay token geometric consistency re-ranking

# --- ALTERED TO SAVE LIST OF PR SUCCESS/FAILURES --- #

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import tempfile
import argparse
import logging
import torch
import time
import tqdm
from typing import Sequence, List, Dict, Optional

from dataset.dataset_utils import (
    make_eval_dataloader,
    make_eval_dataloader_reranking,
    make_eval_dataloader_6DOF,
)
from dataset.augmentation import Normalize
from eval.utils import get_query_database_splits
from eval.sgv.sgv_utils import sgv_parallel
from eval.geotransformer_utils import compute_geotransformer_metrics, compute_registration_error
from eval.egonn_utils import ransac_fn
from misc.logger import create_logger
from misc.point_clouds import icp
from misc.torch_utils import set_seed, to_device, release_cuda
from misc.utils import TrainingParams, load_pickle, save_pickle
from models.model_factory import model_factory
from models.hotformerloc_metric_loc import HOTFormerMetricLoc
from models.hotformerloc_metric_loc_legacy import HOTFormerMetricLoc as HOTFormerMetricLocLegacy
from models.egonn import MinkGL as EgoNN

DISABLE_ICP = False
EVAL_MODES = ['Initial', 'Re-Ranked']
MAX_NN_EUCLID_DIST = 30.0  # max allowable distance for initial NN to be when logging re-ranking failures (WP submap diam is 60m)
MAX_RR_TO_NN_EUCLID_DIST = 10.0  # max distance threshold in metres from nn before re-ranking failure is logged
HIT_MISS_PATH = './hit_miss_v2'

def evaluate(
    model: torch.nn.Module,
    device,
    params: TrainingParams,
    log: bool = False,
    model_name: str = 'model',
    radius: Sequence[float] = [5., 20.],
    icp_refine: bool = False,
    local_max_eval_threshold: float = np.inf,
    num_neighbors: int = 20,
    show_progress: bool = False,
    only_global: bool = False,
    use_ransac: bool = False,
    save_embeddings: bool = False,
    load_embeddings: bool = False,
    reranking: bool = False,
):
    # Run evaluation on all eval datasets
    eval_database_files, eval_query_files = get_query_database_splits(params)

    assert len(eval_database_files) == len(eval_query_files)
    global_metrics, local_metrics = {}, {}
    average_global_metrics, average_local_metrics = {}, {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
        if 'AboveUnder' in params.dataset_name or 'CSWildPlaces' in params.dataset_name:
            # if "pickles/" in database_file:  # CS-WildPlaces
            location_name = database_file.split('/')[-1].split('_')[1]
            temp = query_file.split('/')[-1].split('_')[1]
            # else:
            #     location_name = database_file.split('_')[1]
            #     temp = query_file.split('_')[1]
        else:
            location_name = database_file.split('_')[0]
            temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        if show_progress:
            logging.info(f'Evaluating: {location_name}')
        temp_global_metrics, temp_local_metrics = evaluate_dataset(
            model,
            device,
            params,
            database_sets,
            query_sets,
            location_name,
            log=log,
            model_name=model_name,
            radius=radius,
            icp_refine=icp_refine,
            local_max_eval_threshold=local_max_eval_threshold,
            num_neighbors=num_neighbors,
            show_progress=show_progress,
            only_global=only_global,
            use_ransac=use_ransac,
            save_embeddings=save_embeddings,
            load_embeddings=load_embeddings,
            reranking=reranking,
        )
        global_metrics[location_name] = temp_global_metrics
        average_global_metrics[location_name] = temp_global_metrics['average']
        if not only_global:
            local_metrics[location_name] = temp_local_metrics
            average_local_metrics[location_name] = temp_local_metrics['average']
        
    # Compute average stats
    global_metrics['average'] = {'average': average_nested_dict(average_global_metrics)}
    if not only_global:
        local_metrics['average'] = {'average': average_nested_dict(average_local_metrics)}
    return global_metrics, local_metrics


def evaluate_dataset(
    model: torch.nn.Module,
    device,
    params: TrainingParams,
    database_sets: List[Dict],
    query_sets: List[Dict],
    location_name: str,
    log: bool = False,
    model_name: str = "model",
    radius: Sequence[float] = [5., 20.],
    icp_refine: bool = False,
    local_max_eval_threshold: float = np.inf,
    num_neighbors: int = 20,
    show_progress: bool = False,
    only_global: bool = False,
    use_ransac: bool = False,
    save_embeddings: bool = False,
    load_embeddings: bool = False,
    reranking: bool = False,
):
    # Run evaluation on a single dataset
    global_metrics, local_metrics = {}, {}

    database_cache_files = []
    database_local_cache_files = []
    database_positions = []
    query_cache_files = []
    query_local_cache_files = []

    model.eval()

    # TODO: Determine why memory usage peaks at ~200GB when computing/loading Venman from disk (even though Kara is larger)
    try:
        if show_progress:
            logging.info(f'{"Loading" if load_embeddings else "Computing"} database embeddings')
        for ii, data_set in enumerate(database_sets):
            global_tmp_fp, local_tmp_fp, temp_positions = None, None, None
            if len(data_set) > 0:
                # Create array of coordinates of all db elements
                temp_positions = np.array([(db_details['northing'], db_details['easting']) for db_details in data_set.values()])
                if load_embeddings:
                    temp_embeddings, temp_local_dict = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'database_{ii}')
                else:
                    temp_embeddings, temp_local_dict = get_latent_vectors(model, data_set, device, params, only_global, reranking, show_progress)
                if save_embeddings:
                    save_embeddings_to_file(temp_embeddings, temp_local_dict, model_name, params.dataset_name, location_name, f'database_{ii}')
                # Always cache to tmp directory
                global_tmp_fp, local_tmp_fp = cache_embeddings(temp_embeddings, temp_local_dict)
            database_positions.append(temp_positions)
            database_cache_files.append(global_tmp_fp)
            database_local_cache_files.append(local_tmp_fp)

        if show_progress:
            logging.info(f'{"Loading" if load_embeddings else "Computing"} query embeddings')
        for jj, data_set in enumerate(query_sets):
            global_tmp_fp, local_tmp_fp = None, None
            if len(data_set) > 0:
                if load_embeddings:
                    temp_embeddings, temp_local_dict = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'query_{jj}')
                else:
                    temp_embeddings, temp_local_dict = get_latent_vectors(model, data_set, device, params, only_global, reranking, show_progress)
                if save_embeddings:
                    save_embeddings_to_file(temp_embeddings, temp_local_dict, model_name, params.dataset_name, location_name, f'query_{jj}')
                # Always cache to tmp directory
                global_tmp_fp, local_tmp_fp = cache_embeddings(temp_embeddings, temp_local_dict)
            query_cache_files.append(global_tmp_fp)
            query_local_cache_files.append(local_tmp_fp)

        del temp_embeddings, temp_local_dict

        # Create hit/miss dir
        hit_miss_dir = os.path.join(HIT_MISS_PATH, params.dataset_name, location_name, model_name)
        os.makedirs(hit_miss_dir, exist_ok=True)

        if show_progress:
            logging.info('Running evaluation')
        for ii in range(len(database_sets)):
            # Load cached embeddings
            temp_database_embeddings, temp_database_local_dict = (
                load_cached_embeddings(database_cache_files[ii], database_local_cache_files[ii])
            )
            for jj in range(len(query_sets)):
                if (ii == jj and params.skip_same_run) or database_cache_files[ii] is None or query_cache_files[jj] is None:
                    continue
                if 'CSCampus3D' in params.dataset_name:
                    # For Campus3D, we report on the aerial-only database, which is idx 1
                    if ii != 1:
                        continue
                    split_name = os.path.split(os.path.split(database_sets[ii][0]['query'])[0])[0] + f'_idx{ii}'
                elif params.dataset_name == 'WildPlaces':
                    # For WildPlaces, there are multiple databases per query set, so add both to split name
                    split_name = (os.path.split(os.path.split(database_sets[ii][0]['query'])[0])[0]
                                + '-' + query_sets[jj][0]['query'].split('/')[1])
                else:
                    split_name = os.path.split(os.path.split(query_sets[jj][0]['query'])[0])[0]
                temp_query_embeddings, temp_query_local_dict = (
                    load_cached_embeddings(query_cache_files[jj], query_local_cache_files[jj])
                )
                temp_global_metrics, temp_local_metrics = get_metrics(
                    m=ii,
                    n=jj,
                    database_global_embeddings=temp_database_embeddings,
                    query_global_embeddings=temp_query_embeddings,
                    database_local_dict=temp_database_local_dict,
                    query_local_dict=temp_query_local_dict,
                    database_positions=database_positions[ii],
                    database_set=database_sets[ii],
                    query_set=query_sets[jj],
                    model=model,
                    device=device,
                    params=params,
                    radius=radius,
                    icp_refine=icp_refine,
                    local_max_eval_threshold=local_max_eval_threshold,
                    num_neighbors=num_neighbors,
                    log=log,
                    model_name=model_name,
                    show_progress=show_progress,
                    only_global=only_global,
                    use_ransac=use_ransac,
                    reranking=reranking,
                    hit_miss_dir=hit_miss_dir,
                )
                # Report per-split metrics
                global_metrics[split_name] = temp_global_metrics
                if not only_global:
                    local_metrics[split_name] = temp_local_metrics
                del temp_query_embeddings, temp_query_local_dict
            del temp_database_embeddings, temp_database_local_dict

        # Compute average for split
        global_metrics['average'] = average_nested_dict(global_metrics)
        if not only_global:
            local_metrics['average'] = average_nested_dict(local_metrics)

    finally:
        # Close all temporary files
        all_cache_files = [
            *database_cache_files,
            *database_local_cache_files,
            *query_cache_files,
            *query_local_cache_files,
        ]
        for tmp_fp in all_cache_files:
            if tmp_fp is None:
                continue
            tmp_fp.close()

    return global_metrics, local_metrics

def average_nested_dict(nested_dict: Dict):
    """
    Returns the average values for all keys in a nested dictionary.  
    E.g. {'sample1': {'metric1': {'submetric1': val1}}, 'sample2': {'metric1': {'submetric1': val2}}, ...}
    returns {'metric1': {'submetric1': np.mean([val1, val2, ...]), ...}}
    """
    assert isinstance(nested_dict, dict)
    average_dict = {}
    if len(nested_dict) == 0:
        return average_dict
    sub_dicts = list(nested_dict.values())
    assert all([isinstance(sub_dict_ii, dict) for sub_dict_ii in sub_dicts])
    # Remove any metrics with 'failure' in the key (e.g. rr_failures, failure_query...)
    metrics = [metric for metric in sub_dicts[0].keys() if 'failure' not in str(metric)]
    sub_metrics = [sub_metric for sub_metric in sub_dicts[0][metrics[0]].keys() if 'failure' not in str(sub_metric)]
    for metric_ii in metrics:
        average_dict[metric_ii] = {}
        temp_metric_list = {key: [] for key in sub_metrics}
        for sub_dict_jj in sub_dicts:
            assert metric_ii in sub_dict_jj, 'Invalid nested dict, child keys must match'
            for sub_metric_kk in sub_metrics:
                temp_metric_list[sub_metric_kk].append(sub_dict_jj[metric_ii][sub_metric_kk])
        for sub_metric_kk in sub_metrics:
            average_dict[metric_ii][sub_metric_kk] = np.mean(temp_metric_list[sub_metric_kk], axis=0)
    return average_dict

def get_latent_vectors(
    model: torch.nn.Module,
    data_set: Dict,
    device,
    params: TrainingParams,
    only_global: bool = False,
    reranking: bool = False,
    show_progress: bool = False,
):
    # Adapted from original PointNetVLAD code
    if len(data_set) == 0:
        return None, None

    ### NOTE: Comment out below to test eval during training debug mode
    if params.debug:
        global_embeddings = np.random.randn(len(data_set), params.model_params.output_dim)
        local_dict = {'local_embeddings': []}
        if not only_global:
            if isinstance(model, EgoNN):
                local_dict['keypoints'] = []
                for _ in range(len(data_set)):
                    local_dict['local_embeddings'].append(torch.randn(128, 128))
                    local_dict['keypoints'].append(torch.randn(128, 3))
            elif params.load_octree:
                if reranking:
                    raise NotImplementedError
                # Generate random feats for all possible depths
                start_depth = params.octree_depth - params.model_params.num_input_downsamples
                end_depth = start_depth - model.hotformerloc_global.backbone.backbone.num_stages
                channels = list(model.hotformerloc_global.backbone.backbone.channels)
                num_octf_levels = model.hotformerloc_global.backbone.backbone.num_octf_levels
                num_pyramid_levels = model.hotformerloc_global.backbone.backbone.num_pyramid_levels
                if len(channels[num_octf_levels:]) == 1:
                    channels[num_octf_levels:] = channels[num_octf_levels:] * num_pyramid_levels
                for _ in range(len(data_set)):
                    temp_dict = {}
                    for jj, depth_j in enumerate(range(start_depth, end_depth, -1)):
                        temp_dict[depth_j] = torch.randn(128, channels[jj])
                    local_dict['local_embeddings'].append(temp_dict)
            else:
                raise NotImplementedError
        return global_embeddings, local_dict

    # Create dataloader for data_set
    dataloader = make_eval_dataloader(params, data_set)
    global_embeddings = None
    local_dict = {'local_embeddings': []}
    model.eval()
    with tqdm.tqdm(total=len(dataloader.dataset), disable=(not show_progress)) as pbar:
        for ii, batch_dict in enumerate(dataloader):
            batch = batch_dict['batch']
            batch = to_device(batch, device, non_blocking=True, construct_octree_neigh=True)
            temp_global_embedding, temp_local_dict = compute_embedding(model, batch, only_global, reranking)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(data_set), temp_global_embedding.shape[1]), dtype=temp_global_embedding.dtype)
            global_embeddings[ii*params.val_batch_size:(ii*params.val_batch_size + len(temp_global_embedding))] = temp_global_embedding
            # Split local embeddings from batch
            if 'local_embedding' in temp_local_dict:
                local_dict['local_embeddings'].extend(temp_local_dict['local_embedding'])
            if 'keypoints' in temp_local_dict:
                if ii == 0:
                    local_dict['keypoints'] = []
                local_dict['keypoints'].extend(temp_local_dict['keypoints'])
            pbar.update(len(temp_global_embedding))
    
    return global_embeddings, local_dict


def compute_embedding(
    model: torch.nn.Module,
    batch: Dict,
    only_global: bool = False,
    reranking: bool = False,
):
    with torch.inference_mode():
        # Compute global descriptor
        y = model(batch, global_only=True)
        global_embedding = release_cuda(y['global'], to_numpy=True)
        local_dict = {}
        if (not reranking) and only_global:
            # Don't need local embeddings if not reranking or doing metric loc
            pass
        # Get local descriptors for each pyramid level
        elif 'local' in y:
            local_embeddings = y['local']  # keep as tensors for future forward pass
            if isinstance(model, (HOTFormerMetricLoc, HOTFormerMetricLocLegacy)):
                octree = y['octree']
                local_embeddings_list = [{} for _ in range(octree.batch_size)]
                # Batch stored in concat mode, so need to split back to batch elems
                for depth_j in local_embeddings.keys():
                    batch_lengths_depth_j = octree.batch_nnum_nempty[depth_j].tolist()
                    batch_embeddings_depth_j = local_embeddings[depth_j].split(batch_lengths_depth_j)
                    for ii, embedding in enumerate(batch_embeddings_depth_j):
                        local_embeddings_list[ii][depth_j] = embedding
                local_dict['local_embedding'] = release_cuda(local_embeddings_list)
            else:
                local_dict['local_embedding'] = release_cuda(local_embeddings)
        elif 'keypoints' in y:  # EgoNN
            # Sort by saliency
            descriptors, keypoints, sigma = y['descriptors'], y['keypoints'], y['sigma']
            for ii in range(len(descriptors)):
                n_kpts = min(len(sigma[ii]), 128)  # 128 keypoints by default
                _, indices = torch.topk(sigma[ii].squeeze(1), dim=0, k=n_kpts, largest=False)
                descriptors[ii] = descriptors[ii][indices]
                keypoints[ii] = keypoints[ii][indices]
            local_dict['local_embedding'] = release_cuda(descriptors)
            local_dict['keypoints'] = release_cuda(keypoints)
        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return global_embedding, local_dict


def get_metrics(
    m: int,
    n: int,
    database_global_embeddings: np.ndarray,
    query_global_embeddings: np.ndarray,
    database_local_dict: Dict,
    query_local_dict: Dict,
    database_positions: np.ndarray,
    database_set: Dict[int, Dict],
    query_set: Dict[int, Dict],
    model: torch.nn.Module,
    device: str,
    params: TrainingParams,
    radius: Sequence[float] = [5., 20.],
    icp_refine: bool = False,
    local_max_eval_threshold: float = np.inf,
    num_neighbors: int = 20,
    log: bool = False,
    model_name: str = "model",
    show_progress: bool = False,
    only_global: bool = False,
    use_ransac: bool = False,
    reranking: bool = False,
    hit_miss_dir: str = './',
):
    # ### TEMP FOR DEBUGGING ###
    # return np.ones(25, np.float32), 1.0, 1.0
    # ##########################

    # Determine if using unfair RTE and RRE protocol
    unfair_rre_rte = False
    try:
        unfair_rre_rte = getattr(args, 'unfair_rte_rre', False)
    except NameError:  # if called from trainer.py, args will not exist
        pass

    # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
    global_metrics = {
        'rr_failures': [],
    } if reranking else {}
    intermediate_metrics = {
        'tp': {r: [0] * num_neighbors for r in radius},
        'opr': {r: 0 for r in radius},
        'RR': {r: [] for r in radius},
    }
    if reranking:
        intermediate_metrics.update({
        'tp_rr': {r: [0] * num_neighbors for r in radius},
        'opr_rr': {r: 0 for r in radius},
        'RR_rr': {r: [] for r in radius},
        't_rr': [],
    })
    local_metrics = {}
    if not only_global:
        for eval_mode in EVAL_MODES:
            if not reranking and eval_mode == 'Re-Ranked':
                continue
            local_metrics[eval_mode] = {
                'success': [],
                'rre': [],
                'rte': [],
                'coarse_IR': [],
                'fine_IR': [],
                'fine_overlap': [],
                'fine_residual': [],
                'fine_num_corr': [],
                'num_pts_per_patch': [],
                'num_corr_patches_lgr': [],
                'num_corr_pts_per_patch_lgr': [],
                'corr_score_lgr': [],
                'failure_query_pos_ndx': [],
                't_metloc': [],
            }
            if use_ransac:
                local_metrics[eval_mode].update({
                'success_ransac': [],
                'rre_ransac': [],
                'rte_ransac': [],
                't_ransac': [],
                })
            if icp_refine:
                local_metrics[eval_mode].update({
                'success_refined': [],
                'rre_refined': [],
                'rte_refined': [],
                'failure_query_pos_ndx_refined': [],
                't_metloc_refined': []
                })

    database_global_output = database_global_embeddings
    queries_global_output = query_global_embeddings
    assert all([x is not None for x in [database_global_output, queries_global_output]])

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_global_nbrs = KDTree(database_global_output)

    metric_loc_pairs_list = {eval_mode: [] for eval_mode in EVAL_MODES}

    # NOTE: Recall@1% will be incorrect when `opr_threshold` is > `num_neighbours`
    opr_threshold = max(int(round(len(database_global_output)/100.0)), 1)

    num_evaluated = 0
    global_result_dict = {
        'query_nn_list': [],
        'euclid_dist_list': [],
    }
    hit_miss_str = 'Query idx,Easting,Northing,Initial idx,Initial Success,Initial Easting,Initial Northing,Re-Rank idx,Re-Rank Success,Re-Rank Easting,Re-Rank Northing'
    for query_idx in tqdm.tqdm(range(len(queries_global_output)),
                               desc='Place Recognition',
                               disable=(not show_progress)):
        query_metadata = query_set[query_idx]  # {'query': path, 'northing': , 'easting': , 'pose': }
        query_position = np.array((query_metadata['northing'], query_metadata['easting']))
        if m in query_metadata:  # old tuples store true neighbours directly
            true_neighbors = query_metadata[m]
            if len(true_neighbors) == 0:
                continue
        else:  # expected that new tuples filter queries, but do here just in case
            min_neighbor_dist = np.linalg.norm(query_position - database_positions, axis=-1).min()
            if min_neighbor_dist > min(radius):
                continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, nn_indices = database_global_nbrs.query(  # (1, k) arrays
            queries_global_output[query_idx][None, :], k=num_neighbors,
        )
        distances, nn_indices = distances[0], nn_indices[0]  # (k,) arrays

        # Euclidean distance between the query and nn
        delta = query_position - database_positions[nn_indices]  # (k, 2) array
        euclid_dist = np.linalg.norm(delta, axis=-1)  # (k,) array

        ########################################################################
        # TODO: Compute stats needed for PR curve and F1Max (just using euclid dist)
        # # Find top-1 candidate (via cosine distance).
        # embed_cdist = cdist(query_embedding.reshape(1, -1), self.map_embeddings,
        #                     metric='cosine').reshape(-1)
        # min_dist, nearest_idx = np.min(embed_cdist), np.argmin(embed_cdist)
        # place_candidate = map_positions[nearest_idx]
        # p_dist = np.linalg.norm(query_pos - place_candidate)

        # if min_dist < min_min_dist:
        #     min_min_dist = min_dist
        # if min_dist > max_min_dist:
        #     max_min_dist = min_dist

        # # Evaluate top-1 candidate for PR curve
        # false_positive_threshold = 20  # metres, 20 is consistent with LoGG3D-Net
        # for r in self.radius:
        #     for thres_idx in range(self.num_thresholds):
        #         threshold = cd_thresholds[thres_idx]

        #         if min_dist < threshold:  # Positive Prediction
        #             if p_dist <= r:
        #                 global_metrics['num_true_positive'][r][thres_idx] += 1

        #             elif p_dist > false_positive_threshold:
        #                 global_metrics['num_false_positive'][r][thres_idx] += 1

        #         else:  # Negative Prediction
        #             if p_dist > r:
        #                 global_metrics['num_true_negative'][r][thres_idx] += 1
        #             else:
        #                 global_metrics['num_false_negative'][r][thres_idx] += 1
        ########################################################################

        if log:
            # Log false positives (returned as the first element)
            # Check if there's a false positive returned as the first element
            if nn_indices[0] not in true_neighbors:
                fp_ndx = nn_indices[0]
                fp = database_set[fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0]  # Distance in embedding space
                fp_world_dist = np.sqrt((query_metadata['northing'] - fp['northing']) ** 2 +
                                        (query_metadata['easting'] - fp['easting']) ** 2)
                # Find the first true positive
                tp = None
                for k in range(len(nn_indices)):
                    if nn_indices[k] in true_neighbors:
                        closest_pos_ndx = nn_indices[k]
                        tp = database_set[closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                        tp_emb_dist = distances[k]
                        tp_world_dist = np.sqrt((query_metadata['northing'] - tp['northing']) ** 2 +
                                                (query_metadata['easting'] - tp['easting']) ** 2)
                        break
                            
                out_fp_file_name = f"{model_name}_log_fp.txt"
                with open(out_fp_file_name, "a") as f:
                    s = "{}, {}, {:0.2f}, {:0.2f}".format(query_metadata['query'], fp['query'], fp_emb_dist, fp_world_dist)
                    if tp is None:
                        s += ', 0, 0, 0\n'
                    else:
                        s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
                    f.write(s)

            # Save details of 5 best matches for later visualization for 1% of queries
            s = f"{query_metadata['query']}, {query_metadata['northing']}, {query_metadata['easting']}"
            for k in range(min(len(nn_indices), 5)):
                is_match = nn_indices[k] in true_neighbors
                e_ndx = nn_indices[k]
                e = database_set[e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
                e_emb_dist = distances[k]
                world_dist = np.sqrt((query_metadata['northing'] - e['northing']) ** 2 +
                                        (query_metadata['easting'] - e['easting']) ** 2)
                s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
            s += '\n'
            out_top5_file_name = f"{model_name}_log_search_results.txt"
            with open(out_top5_file_name, "a") as f:
                f.write(s)

        ########################################################################
        # # Re-ranking with SGV
        # # NOTE: TO DO THIS, NEED TO PRE-COMPUTE THE COARSE CENTROIDS, OR GET THEM AFTER RUNNING FORWARD PASS OF HOTFORMERMETRICLOC
        if reranking:
            assert num_neighbors >= params.rerank_num_neighbours, 'Set num_neighbours higher'
            global_result_dict['query_nn_list'].append((query_idx, nn_indices[:params.rerank_num_neighbours]))
            global_result_dict['euclid_dist_list'].append(euclid_dist)
            
            # if params.model_params.rerank_mode == 'sgv':
            #     # topk = min(num_neighbors, len(nn_indices))
            #     # tick = time.perf_counter()
            #     # candidate_local_embeddings = database_local_embeddings[m][nn_indices]
            #     # candidate_keypoints = local_map_embeddings_keypoints[m][nn_indices]
            #     # fitness_list = sgv_fn(query_local_embeddings[n][query_idx], candidate_local_embeddings, candidate_keypoints, d_thresh=0.4)
            #     # topk_rerank = np.flip(np.asarray(fitness_list).argsort())
            #     # topk_rerank_indices = copy.deepcopy(nn_indices)
            #     # topk_rerank_indices[:topk] = nn_indices[topk_rerank]
            #     # t_rerank = time.perf_counter() - tick
            #     # intermediate_metrics['t_rr'].append(t_rerank)
            #     raise NotImplementedError

            # # Re-ranking with relay token geometric consistency 
            # elif params.model_params.rerank_mode == 'relay_token_gc':
            #     pass
            #     ####################################################################
            #     # THIS BLOCK CONTAINS THE SIMPLE BATCH CREATION APPROACH FOR
            #     # RE-RANKING, i.e. DIRECTLY LOADING ALL OCTREES AND RUNNING A NEW
            #     # (WASTED) FORWARD PASS. CURENTLY WORKS BUT IS SLOW. REPLACING WITH
            #     # A TORCH DATALOADER THAT DOES THE SAME THING.
            #     ####################################################################
            #     # rerank_batch_temp = []
            #     # rerank_batch_temp.append(query_dataset[query_idx])
            #     # for nn_idx in nn_indices:
            #     #     rerank_batch_temp.append(database_dataset[nn_idx])
            #     # rerank_batch_dict = to_device(eval_collate_fn(rerank_batch_temp), device, construct_octree_neigh=True)
            #     # rerank_batch = rerank_batch_dict['batch']
            #     # rerank_shift_and_scale = rerank_batch_dict['shift_and_scale']
            #     # with torch.inference_mode():
            #     #     out = model(rerank_batch, global_only=True)
            #     #     rerank_scores = model.rerank_inference(out, rerank_shift_and_scale)
            #     #     topk_rerank, topk_rerank_indices = torch.sort(rerank_scores, dim=1, descending=True)
            #     ####################################################################

            #     ####################################################################
            #     # THIS BLOCK CONTAINS INITIAL ATTEMPTS TO PRE-COMPUTE RELAY TOKENS
            #     # AND COMBINE THEM INTO NEW BATCHES FOR RE-RANKING. UNFINISHED.
            #     ####################################################################
            #     # NOTE: Currently I don't think this is feasible to do, due to the
            #     #       way that the number of relay tokens for a given batch element
            #     #       will vary depending on the length of it's neighbour elements.
            #     #       Simple solution is just to run the forward pass again, which is
            #     #       slow, but is guaranteed to output RTs in the right format.
            #     # rerank_batch_temp, rt_batch, rt_attn_batch = [], [], []
            #     # rerank_batch_temp.append(query_dataset[query_idx])
            #     # rt_batch.append(query_local_dict['rt'][query_idx])
            #     # rt_attn_batch.append(query_local_dict['rt_final_cls_attn_vals'][query_idx])
            #     # for nn_idx in nn_indices:
            #     #     rerank_batch_temp.append(database_dataset[nn_idx])
            #     #     rt_batch.append(database_local_dict['rt'][nn_idx])
            #     #     rt_attn_batch.append(database_local_dict['rt_final_cls_attn_vals'][nn_idx])
            #     # rerank_batch_dict = to_device(eval_collate_fn(rerank_batch_temp), device, construct_octree_neigh=True)
            #     # rerank_batch = rerank_batch_dict['batch']
            #     # rerank_shift_and_scale = rerank_batch_dict['shift_and_scale']
            #     # # Create OctreeT
            #     # if 'hotformerloc' in model_name.lower():
            #     #     octree = model.backbone.backbone.construct_OctreeT(
            #     #         rerank_batch, depth=(params.octree_depth-params.model_params.num_input_downsamples)
            #     #     )
            #     # elif 'hotformermetricloc' in model_name.lower():
            #     #     octree = model.hotformerloc_global.backbone.backbone.construct_OctreeT(
            #     #         rerank_batch, depth=(params.octree_depth-params.model_params.num_input_downsamples)
            #     #     )
            #     # else:
            #     #     raise NotImplementedError
            #     # query_rt = query_local_dict['rt'][query_idx]
            #     # nn_rt_list = [database_local_dict['rt'][nn_idx] for nn_idx in nn_indices]
            #     # concat_and_pad_rt(out['rt'], octree, pad=False, remove_final_padding=True)
            #     # unpad_and_split_rt()
            #     # TODO: Pass through model
            #     # rerank_scores = model.rerank_inference(rerank_batch, rerank_shift_and_scale)
            #     # topk_rerank, topk_rerank_indices = torch.sort(rerank_scores, descending=True)
            #     ####################################################################

            # elif params.model_params.rerank_mode is not None:
            #     raise NotImplementedError

            #     # delta_rerank = query_position - database_positions[m][topk_rerank_indices]
            #     # euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)

            #     # # Log cases where re-ranking is worse (causes PR failure, or is
            #     # #   significantly worse than the original top-candidate)
            #     # rr_to_nn_euclid_dist = euclid_dist_rr - euclid_dist
            #     # if (euclid_dist <= self.MAX_NN_EUCLID_DIST
            #     #         and rr_to_nn_euclid_dist > self.MAX_RR_TO_NN_EUCLID_DIST):
            #     #     # print(f'Fail: {euclid_dist_rr:.2f}m > {euclid_dist:.2f}m', flush=True)
            #     #     query_name = os.path.basename(self.eval_set.query_set[query_idx].rel_scan_filepath)
            #     #     nn_name = os.path.basename(self.eval_set.map_set[nn_indices].rel_scan_filepath)
            #     #     nn_rerank_name = os.path.basename(self.eval_set.map_set[topk_rerank_indices].rel_scan_filepath)
            #     #     global_metrics['rr_failures'].append((query_name, nn_name,
            #     #                                             nn_rerank_name,
            #     #                                             f'{euclid_dist:.2f}',
            #     #                                             f'{euclid_dist_rr:.2f}'))
            # ########################################################################

        # Count true positives and 1% retrieved for different radius and NN number
        intermediate_metrics['tp'] = {r: [intermediate_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in radius}
        intermediate_metrics['opr'] = {r: intermediate_metrics['opr'][r] + (1 if (euclid_dist[:opr_threshold] <= r).any() else 0) for r in radius}
        intermediate_metrics['RR'] = {r: intermediate_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in radius}
        # NOTE: rr metrics now handled in below loop
        # intermediate_metrics['tp_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in self.radius}
        # intermediate_metrics['opr_rr'] = {r: intermediate_metrics['opr_rr'][r] + (1 if (euclid_dist_rr[:threshold] <= r).any() else 0) for r in self.radius}
        # intermediate_metrics['RR_rr'] = {r: intermediate_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}
        if only_global or reranking:
            continue

        # LOCAL DESCRIPTOR EVALUATION
        # Do the evaluation only if the nn pose is within distance threshold
        # Otherwise the overlap is too small to get reasonable results
        if euclid_dist[0] > local_max_eval_threshold:
            continue

        # Cache query and nn idx for metric loc eval (if not considering re-ranking)
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
                                                desc='Re-ranking',
                                                disable=(not show_progress)):
            if params.debug and idx >= 2:
                break
            query_idx = global_result_dict['query_nn_list'][idx][0]
            nn_indices = global_result_dict['query_nn_list'][idx][1]
            euclid_dist = global_result_dict['euclid_dist_list'][idx]
            query_metadata = query_set[query_idx]  # {'query': path, 'northing': , 'easting': , 'pose': }
            query_position = np.array((query_metadata['northing'], query_metadata['easting']))

            # Separate forward pass for EgoNN+SGV
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
                tic_rr = time.perf_counter()
                rerank_scores = torch.tensor(sgv_parallel(
                    src_keypts=query_keypoints,
                    tgt_keypts=candidate_keypoints,
                    src_features=query_features,
                    tgt_features=candidate_features,
                    d_thresh=params.sgv_d_thresh,
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
                    tic_rr = time.perf_counter()
                    if params.model_params.rerank_mode == 'sgv':
                        rerank_dict = model.sgv_rerank_inference(
                            model_out=out_dict,
                            shift_and_scale=rerank_shift_and_scale,
                            batch=rerank_batch,
                            feat_type=sgv_feat_type,
                            d_thresh=params.sgv_d_thresh,
                        )
                        rerank_scores = rerank_dict['scores']
                    else:
                        rerank_dict = model.rerank_inference(
                            model_out=out_dict,
                            shift_and_scale=rerank_shift_and_scale,
                            batch=rerank_batch,
                        )
                        rerank_scores = rerank_dict['scores'][0, :, 0]

            intermediate_metrics['t_rr'].append(time.perf_counter() - tic_rr)
            _, rerank_sort_indices = release_cuda(
                torch.sort(rerank_scores, descending=True), to_numpy=True
            )
            topk_rerank_indices = nn_indices[rerank_sort_indices]
            delta_rerank = query_position - database_positions[topk_rerank_indices]
            euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)

            # Log cases where re-ranking is worse (causes PR failure, or is
            #   significantly worse than the original top-candidate)
            rr_to_nn_euclid_dist = euclid_dist_rr[0] - euclid_dist[0]
            if (
                euclid_dist[0] <= MAX_NN_EUCLID_DIST
                and rr_to_nn_euclid_dist > MAX_RR_TO_NN_EUCLID_DIST
            ):
                # print(f'Fail: {euclid_dist_rr[0]:.2f}m > {euclid_dist[0]:.2f}m', flush=True)
                query_name = os.path.basename(query_set[query_idx]['query'])
                query_name += f' ({query_idx})'
                nn_name = os.path.basename(database_set[nn_indices[0]]['query'])
                nn_name += f' ({nn_indices[0]})'
                nn_rerank_name = os.path.basename(database_set[topk_rerank_indices[0]]['query'])
                nn_rerank_name += f' ({topk_rerank_indices[0]})'
                global_metrics['rr_failures'].append(
                    (query_name, nn_name, nn_rerank_name,
                        f'{euclid_dist[0]:.2f}', f'{euclid_dist_rr[0]:.2f}')
                )

            # Save initial and re-ranking success/failures
            init_success = euclid_dist[0] <= radius[-1]  # use larger retrieval threshold
            rr_success = euclid_dist_rr[0] <= radius[-1]
            hit_miss_str += f'\n{query_idx},{query_metadata['easting']},{query_metadata['northing']}'
            hit_miss_str += f',{nn_indices[0]},{int(init_success)},{database_set[nn_indices[0]]['easting']},{database_set[nn_indices[0]]['northing']}'
            hit_miss_str += f',{topk_rerank_indices[0]},{int(rr_success)},{database_set[topk_rerank_indices[0]]['easting']},{database_set[topk_rerank_indices[0]]['northing']}'

            intermediate_metrics['tp_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in radius}
            intermediate_metrics['opr_rr'] = {r: intermediate_metrics['opr_rr'][r] + (1 if (euclid_dist_rr[:opr_threshold] <= r).any() else 0) for r in radius}
            intermediate_metrics['RR_rr'] = {r: intermediate_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in radius}
            if only_global:
                continue

            # LOCAL DESCRIPTOR EVALUATION
            # Do the evaluation only if the nn pose is within distance threshold
            # Otherwise the overlap is too small to get reasonable results
            # (evaluation continues only if standard AND re-ranked nn pose is within thresh meters,
            #  to ensure both are evaluated on the same set of queries)
            if euclid_dist[0] > local_max_eval_threshold or euclid_dist_rr[0] > local_max_eval_threshold:
                continue

            # Cache query and nn idx for metric loc eval
            metric_loc_pairs_list['Initial'].append((query_idx, nn_indices[0]))
            metric_loc_pairs_list['Re-Ranked'].append((query_idx, topk_rerank_indices[0]))

    if isinstance(model, EgoNN):
        metric_loc_func = metric_loc_egonn
    else:
        metric_loc_func = metric_loc_hotformerloc

    # Run metric localisation evaluation for initial and re-ranked top-candidate
    for eval_mode in EVAL_MODES:
        if only_global:
            break
        if not reranking and eval_mode == 'Re-Ranked':
            continue
        eval_dataloader = make_eval_dataloader_6DOF(
            params, query_set, database_set, metric_loc_pairs_list[eval_mode]
        )
        # TODO: Process with batch size > 1 to reduce dataloader bottleneck
        for idx, batch in tqdm.tqdm(enumerate(eval_dataloader),
                                    total=len(eval_dataloader),
                                    desc=f'Metric Localisation [{eval_mode}]',
                                    disable=(not show_progress)):
            if params.debug and idx >= 2:
                break
            local_metrics = metric_loc_func(
                batch=batch,
                idx=idx,
                eval_mode=eval_mode,
                metric_loc_pairs_list=metric_loc_pairs_list,
                local_metrics=local_metrics,
                query_local_dict=query_local_dict,
                database_local_dict=database_local_dict,
                model=model,
                device=device,
                params=params,
                icp_refine=icp_refine,
                unfair_rre_rte=unfair_rre_rte,
                use_ransac=use_ransac,
            )

    # Save hit/miss to file
    hit_miss_file = os.path.join(hit_miss_dir, f'db{m}_q{n}_hits.csv')
    with open(hit_miss_file, 'w') as f:
        f.write(hit_miss_str)

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
        global_metrics['mean_t_rr'] = {r: np.mean(np.asarray(intermediate_metrics['t_rr'])) for r in radius}  # duplicating for each radius so `average_nested_dict` doesn't break

    mean_local_metrics = {}
    if not only_global:
        # Calculate mean values of local descriptor metrics
        for eval_mode in EVAL_MODES:
            if not reranking and eval_mode == 'Re-Ranked':
                continue
            mean_local_metrics[eval_mode] = {}
            for metric in local_metrics[eval_mode]:
                m_l = local_metrics[eval_mode][metric]
                if len(m_l) == 0:
                    mean_local_metrics[eval_mode][metric] = 0.
                    if 't_metloc' in metric:
                        mean_local_metrics[eval_mode][f'{metric}_sd'] = 0.
                else:
                    if 'failure_query_pos_ndx' in metric:  # we want a list of all query + pos failure pairs
                        mean_local_metrics[eval_mode][metric] = m_l
                        continue
                    if 't_metloc' in metric:
                        mean_local_metrics[eval_mode][f'{metric}_sd'] = np.std(m_l)
                    elif 't_ransac' in metric:
                        mean_local_metrics[eval_mode][f'{metric}_sd'] = np.std(m_l)
                    mean_local_metrics[eval_mode][metric] = np.mean(m_l)

    ########################################################################
    # TODO: Compute PR curves and F1 max (with and without re-ranking)
    # print(f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
    # for r in self.radius:
    #     F1max = 0.0
    #     Precisions, Recalls = [], []
    #     for thres_idx in range(self.num_thresholds):
    #         nTrueNegative = global_metrics['num_true_negative'][r][thres_idx]
    #         nFalsePositive = global_metrics['num_false_positive'][r][thres_idx]
    #         nTruePositive = global_metrics['num_true_positive'][r][thres_idx]
    #         nFalseNegative = global_metrics['num_false_negative'][r][thres_idx]

    #         Precision = 0.0
    #         Recall = 0.0
    #         F1 = 0.0

    #         if nTruePositive > 0.0:
    #             Precision = nTruePositive / (nTruePositive + nFalsePositive)
    #             Recall = nTruePositive / (nTruePositive + nFalseNegative)

    #             F1 = 2 * Precision * Recall * (1/(Precision + Recall))

    #         if F1 > F1max:
    #             F1max = F1
    #             F1_TN = nTrueNegative
    #             F1_FP = nFalsePositive
    #             F1_TP = nTruePositive
    #             F1_FN = nFalseNegative
    #             F1_thresh_id = thres_idx
    #         Precisions.append(Precision)
    #         Recalls.append(Recall)
    #     print(f'Radius: {r} [m]:')
    #     print(f'  F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}')
    #     print(f'  F1_thresh_id: {F1_thresh_id}')
    #     print(f'  F1_thresh: {cd_thresholds[F1_thresh_id]:.4f}')
    #     print(f'  F1max: {F1max:.4f}')
        
    #     if self.save_pr_values:
    #         self.save_pr_values_to_file(Precisions, Recalls, r)

    #     if self.plot_pr_curve:
    #         plt.title('Seq: ' + self.eval_seq +
    #                     '    F1Max: ' + "%.4f" % (F1max))
    #         plt.plot(Recalls, Precisions, marker='.')
    #         plt.xlabel('Recall')
    #         plt.ylabel('Precision')
    #         plt.axis([0, 1, 0, 1.1])
    #         plt.xticks(np.arange(0, 1.01, step=0.1))
    #         plt.grid(True)
    #         save_dir = os.path.join(os.path.dirname(__file__), 'pr_curves', self.eval_seq)
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         # plt.show()
    #         plt.savefig(os.path.join(save_dir, f'model_{self.model_name}_rad{r}m.png'))
    #         plt.close()
    ########################################################################

    return global_metrics, mean_local_metrics

def metric_loc_egonn(
    batch: dict,
    idx: int,
    eval_mode: str,
    metric_loc_pairs_list: Dict[str, list],
    local_metrics: Dict[str, dict],
    query_local_dict: Dict[str, list],
    database_local_dict: Dict[str, list],
    model: torch.nn.Module,
    device: str,
    params: TrainingParams,
    icp_refine: bool,
    unfair_rre_rte: bool,
    n_kpts: int = 128,
    **kwargs,
):
    query_idx, nn_idx = metric_loc_pairs_list[eval_mode][idx]
    T_gt = batch['transform'][0].numpy()

    # Use pre-computed embeddings so we don't re-compute the entire forward pass
    query_keypoints = query_local_dict['keypoints'][query_idx]
    candidate_keypoints = database_local_dict['keypoints'][nn_idx]
    query_features = query_local_dict['local_embeddings'][query_idx]
    candidate_features = database_local_dict['local_embeddings'][nn_idx]

    tic = time.perf_counter()
    T_estimated, inliers, fitness = ransac_fn(
        query_keypoints=query_keypoints,
        candidate_keypoints=candidate_keypoints,
        query_features=query_features,
        candidate_features=candidate_features,
        n_k=n_kpts,
    )
    t_metloc = time.perf_counter() - tic
    inlier_ratio = inliers / n_kpts

    # Refine the estimated pose using ICP
    if icp_refine:
        # Get point clouds in metric coordinates
        query_pc_metric = batch['anc_batch']['pcd'][0]
        nn_pc_metric = batch['pos_batch']['pcd'][0]
        if params.normalize_points:
            # TODO: Need to unnormalize EgoNN keypoints too
            raise NotImplementedError
            query_pc_metric = Normalize.unnormalize(query_pc_metric, batch['anc_shift_and_scale'][0])
            nn_pc_metric = Normalize.unnormalize(nn_pc_metric, batch['pos_shift_and_scale'][0])
        query_pc_metric = release_cuda(query_pc_metric, to_numpy=True)
        nn_pc_metric = release_cuda(nn_pc_metric, to_numpy=True)

        tic = time.perf_counter()
        T_estimated_refined, _, _ = icp(
            query_pc_metric,
            nn_pc_metric,
            T_estimated,
            gicp=params.local.icp_use_gicp,
            inlier_dist_threshold=params.local.icp_inlier_dist_threshold,
            max_iteration=params.local.icp_max_iteration,
            voxel_size=params.local.icp_voxel_size,
        )
        t_icp = time.perf_counter() - tic

    # Compute metrics with and without ICP refinement
    rre, rte = compute_registration_error(T_gt, T_estimated)
    success = float(rre < params.local.rre_threshold and rte < params.local.rte_threshold)

    local_metrics[eval_mode]['coarse_IR'].append(inlier_ratio)
    local_metrics[eval_mode]['success'].append(success)
    if unfair_rre_rte and success == 1.0:
        local_metrics[eval_mode]['rre'].append(rre)
        local_metrics[eval_mode]['rte'].append(rte)
    else:
        if not unfair_rre_rte:
            local_metrics[eval_mode]['rre'].append(rre)
            local_metrics[eval_mode]['rte'].append(rte)
    if success == 0:
        local_metrics[eval_mode]['failure_query_pos_ndx'].append((query_idx, nn_idx))
    local_metrics[eval_mode]['t_metloc'].append(t_metloc)  # Metric Loc time

    if icp_refine:
        rre_refined, rte_refined = compute_registration_error(T_gt, T_estimated_refined)
        success_refined = float(rre_refined < params.local.rre_threshold and rte_refined < params.local.rte_threshold)
        local_metrics[eval_mode]['success_refined'].append(success_refined)
        if unfair_rre_rte and success_refined == 1.0:
            local_metrics[eval_mode]['rre_refined'].append(rre_refined)
            local_metrics[eval_mode]['rte_refined'].append(rte_refined)
        else:
            if not unfair_rre_rte:
                local_metrics[eval_mode]['rre_refined'].append(rre_refined)
                local_metrics[eval_mode]['rte_refined'].append(rte_refined)
        if success_refined == 0:
            local_metrics[eval_mode]['failure_query_pos_ndx_refined'].append((query_idx, nn_idx))
        local_metrics[eval_mode]['t_metloc_refined'].append(t_metloc + t_icp)  # Metric Loc Refined time

    return local_metrics

def metric_loc_hotformerloc(
    batch: dict,
    idx: int,
    eval_mode: str,
    metric_loc_pairs_list: Dict[str, list],
    local_metrics: Dict[str, dict],
    query_local_dict: Dict[str, list],
    database_local_dict: Dict[str, list],
    model: torch.nn.Module,
    device: str,
    params: TrainingParams,
    icp_refine: bool,
    unfair_rre_rte: bool,
    use_ransac: bool,
    **kwargs,
):
    query_idx, nn_idx = metric_loc_pairs_list[eval_mode][idx]
    T_gt = batch['transform'][0]

    # Use pre-computed embeddings so we don't re-compute the entire forward pass
    if params.debug:
        batch['anc_local_feats'] = {
            'coarse': torch.randn(batch['anc_batch']['octree'].batch_nnum_nempty[model.depth_coarse], params.model_params.channels[-1]),
            'fine': torch.randn(batch['anc_batch']['octree'].batch_nnum_nempty[model.depth_fine], params.model_params.channels[-3]),
        }
        batch['pos_local_feats'] = {
            'coarse': torch.randn(batch['pos_batch']['octree'].batch_nnum_nempty[model.depth_coarse], params.model_params.channels[-1]),
            'fine': torch.randn(batch['pos_batch']['octree'].batch_nnum_nempty[model.depth_fine], params.model_params.channels[-3]),
        }
    else:
        batch['anc_local_feats'] = query_local_dict['local_embeddings'][query_idx]
        batch['pos_local_feats'] = database_local_dict['local_embeddings'][nn_idx]

    batch = to_device(batch, device, construct_octree_neigh=False)  # only need neighs for HOTFloc forward pass

    with torch.inference_mode():
        tic = time.perf_counter()
        model_out = model(batch)
        t_metloc = time.perf_counter() - tic
        T_estimated = release_cuda(model_out[0]['estimated_transform'], to_numpy=True)
    
    # Refine the estimated pose using ICP
    if icp_refine:
        # Get point clouds in metric coordinates
        query_pc_metric = Normalize.unnormalize(batch['anc_batch']['points'].points, batch['anc_shift_and_scale'][0])
        query_pc_metric = release_cuda(query_pc_metric, to_numpy=True)
        nn_pc_metric = Normalize.unnormalize(batch['pos_batch']['points'].points, batch['pos_shift_and_scale'][0])
        nn_pc_metric = release_cuda(nn_pc_metric, to_numpy=True)

        tic = time.perf_counter()
        T_estimated_refined, _, _ = icp(
            query_pc_metric,
            nn_pc_metric,
            T_estimated,
            gicp=params.local.icp_use_gicp,
            inlier_dist_threshold=params.local.icp_inlier_dist_threshold,
            max_iteration=params.local.icp_max_iteration,
            voxel_size=params.local.icp_voxel_size,
        )
        t_icp = time.perf_counter() - tic

    # Compute metrics with and without ICP refinement
    batch_temp = {'transform': batch['transform'][0]}  # temp fix since loss func expects a single batch item

    model_out_np = release_cuda(model_out[0], to_numpy=True)
    batch_temp_np = release_cuda(batch_temp, to_numpy=True)
    temp_metrics = compute_geotransformer_metrics(
        model_out_np, batch_temp_np, params, use_ransac=use_ransac
    )
    local_metrics[eval_mode]['coarse_IR'].append(temp_metrics['PIR'])
    local_metrics[eval_mode]['fine_IR'].append(temp_metrics['IR'])
    local_metrics[eval_mode]['fine_overlap'].append(temp_metrics['OV'])
    local_metrics[eval_mode]['fine_residual'].append(temp_metrics['residual'])
    local_metrics[eval_mode]['fine_num_corr'].append(temp_metrics['num_corr'])
    local_metrics[eval_mode]['success'].append(temp_metrics['success_lgr'])
    if unfair_rre_rte and temp_metrics['success_lgr'] == 1.0:
        local_metrics[eval_mode]['rre'].append(temp_metrics['rre_lgr'])
        local_metrics[eval_mode]['rte'].append(temp_metrics['rte_lgr'])
    else:
        if not unfair_rre_rte:
            local_metrics[eval_mode]['rre'].append(temp_metrics['rre_lgr'])
            local_metrics[eval_mode]['rte'].append(temp_metrics['rte_lgr'])
    local_metrics[eval_mode]['num_pts_per_patch'].append(temp_metrics['num_pts_per_patch'])
    local_metrics[eval_mode]['num_corr_patches_lgr'].append(temp_metrics['num_corr_patches_lgr'])
    local_metrics[eval_mode]['num_corr_pts_per_patch_lgr'].append(temp_metrics['num_corr_pts_per_patch_lgr'])
    local_metrics[eval_mode]['corr_score_lgr'].append(temp_metrics['corr_score_lgr'])
    if temp_metrics['success_lgr'] == 0:
        local_metrics[eval_mode]['failure_query_pos_ndx'].append((query_idx, nn_idx))
    local_metrics[eval_mode]['t_metloc'].append(t_metloc)  # Metric Loc time
    if use_ransac:
        local_metrics[eval_mode]['success_ransac'].append(temp_metrics['success_ransac'])
        if unfair_rre_rte and temp_metrics['success_ransac'] == 1.0:
            local_metrics[eval_mode]['rre_ransac'].append(temp_metrics['rre_ransac'])
            local_metrics[eval_mode]['rte_ransac'].append(temp_metrics['rte_ransac'])
        else:
            if not unfair_rre_rte:
                local_metrics[eval_mode]['rre_ransac'].append(temp_metrics['rre_ransac'])
                local_metrics[eval_mode]['rte_ransac'].append(temp_metrics['rte_ransac'])
        local_metrics[eval_mode]['t_ransac'].append(temp_metrics['t_ransac'])

    if icp_refine:
        model_out[0]['estimated_transform'] = torch.tensor(
            T_estimated_refined, dtype=T_gt.dtype, device=device
        )
        model_out_np = release_cuda(model_out[0], to_numpy=True)
        temp_metrics_refined = compute_geotransformer_metrics(
            model_out_np, batch_temp_np, params, use_ransac=False,  # RANSAC already computed once
        )
        local_metrics[eval_mode]['success_refined'].append(temp_metrics_refined['success_lgr'])
        if unfair_rre_rte and temp_metrics_refined['success_lgr'] == 1.0:
            local_metrics[eval_mode]['rre_refined'].append(temp_metrics_refined['rre_lgr'])
            local_metrics[eval_mode]['rte_refined'].append(temp_metrics_refined['rte_lgr'])
        else:
            if not unfair_rre_rte:
                local_metrics[eval_mode]['rre_refined'].append(temp_metrics_refined['rre_lgr'])
                local_metrics[eval_mode]['rte_refined'].append(temp_metrics_refined['rte_lgr'])
        if temp_metrics_refined['success_lgr'] == 0:
            local_metrics[eval_mode]['failure_query_pos_ndx_refined'].append((query_idx, nn_idx))
        local_metrics[eval_mode]['t_metloc_refined'].append(t_metloc + t_icp)  # Metric Loc Refined time

    torch.cuda.empty_cache()

    return local_metrics

def print_eval_stats(
    global_metrics: Dict,
    local_metrics: Dict,
    icp_refine=False,
    print_false_positives=False,
    reranking=False,
    rerank_mode: Optional[str] = None,
):
    msg = 'Eval Results'
    for database_name in global_metrics.keys():
        msg += '\nDataset: {}'.format(database_name)
        for split in global_metrics[database_name].keys():
            msg += '\n  Split: {}'.format(split)
            msg += '\n    Initial Retrieval:'
            recall = global_metrics[database_name][split]['recall']
            for radius in recall.keys():
                msg += f"\n      Radius: {radius} [m]:\n        Recall@N: "
                for ii, x in enumerate(recall[radius]):
                        msg += f"{x:0.3f}, "
                        if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                            msg += "\n                  "
                msg += "\n        Recall@1%: {:0.3f}".format(global_metrics[database_name][split]['recall@1%'][radius])
                msg += '\n        MRR: {:0.3f}'.format(global_metrics[database_name][split]['MRR'][radius])

            if reranking:
                msg += f'\n    Re-Ranking ({rerank_mode}):'
                recall_rr = global_metrics[database_name][split]['recall_rr']
                for radius_rr in recall_rr.keys():
                    msg += f"\n      Radius: {radius_rr} [m]:\n        Recall@N: "
                    for ii, x in enumerate(recall_rr[radius_rr]):
                        msg += f"{x:0.3f}, "
                        if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                            msg += "\n                  "
                    msg += "\n        Recall@1%: {:0.3f}".format(global_metrics[database_name][split]['recall@1%_rr'][radius_rr])
                    msg += '\n        MRR: {:0.3f}'.format(global_metrics[database_name][split]['MRR_rr'][radius_rr])
                msg += '\n        Re-Ranking Time: {:0.3f} [ms]'.format(1000.0*global_metrics[database_name][split]['mean_t_rr'][radius_rr])
                if (print_false_positives and 'rr_failures' in global_metrics[database_name][split]
                    and len(global_metrics[database_name][split]['rr_failures']) > 0):
                    msg += '\n        Re-Ranking Failures (query, nn, nn_rerank, nn_dist, nn_rerank_dist):\n          '
                    msg += '\n          '.join([', '.join(x) for x in global_metrics[database_name][split]['rr_failures']])

            if len(local_metrics) == 0:
                msg += '\n'
                continue
            msg += '\n    Metric Localization:'
            for eval_mode in EVAL_MODES:
                if eval_mode not in local_metrics[database_name][split]:
                    continue
                msg += f'\n      Eval Mode: {eval_mode}'
                for metric in local_metrics[database_name][split][eval_mode]:
                    if '_refined' in metric and not icp_refine:
                        continue
                    if 'failure_query_pos_ndx' in metric:
                        if not print_false_positives:
                            continue
                        msg += f"\n      {metric}: {local_metrics[database_name][split][eval_mode][metric]}"
                    else:
                        msg += f"\n      {metric}: {local_metrics[database_name][split][eval_mode][metric]:0.3f}"
            msg += '\n'
        msg += '\n'
    logging.info(msg)

def write_eval_stats(
    file_name: str,
    prefix: str,
    global_metrics: Dict,
    local_metrics: Dict,
    icp_refine=False,
    log_false_positives=False,
    reranking=False,
    rerank_mode: Optional[str] = None,
):
    # Save results on the final model
    msg = prefix
    with open(file_name, 'a') as f:
        for database_name in global_metrics.keys():
            msg += '\nDataset: {}'.format(database_name)
            for split in global_metrics[database_name].keys():
                msg += '\n  Split: {}'.format(split)
                msg += '\n    Initial Retrieval:'
                recall = global_metrics[database_name][split]['recall']
                for radius in recall.keys():
                    msg += f"\n      Radius: {radius} [m]:\n        Recall@N: "
                    for ii, x in enumerate(recall[radius]):
                            msg += f"{x:0.3f}, "
                            if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                                msg += "\n                  "
                    msg += "\n        Recall@1%: {:0.3f}".format(global_metrics[database_name][split]['recall@1%'][radius])
                    msg += '\n        MRR: {:0.3f}'.format(global_metrics[database_name][split]['MRR'][radius])

                if reranking:
                    msg += f'\n    Re-Ranking ({rerank_mode}):'
                    recall_rr = global_metrics[database_name][split]['recall_rr']
                    for radius_rr in recall_rr.keys():
                        msg += f"\n      Radius: {radius_rr} [m]:\n        Recall@N: "
                        for ii, x in enumerate(recall_rr[radius_rr]):
                            msg += f"{x:0.3f}, "
                            if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                                msg += "\n                  "
                        msg += "\n        Recall@1%: {:0.3f}".format(global_metrics[database_name][split]['recall@1%_rr'][radius_rr])
                        msg += '\n        MRR: {:0.3f}'.format(global_metrics[database_name][split]['MRR_rr'][radius_rr])
                    msg += '\n        Re-Ranking Time: {:0.3f} [ms]'.format(1000.0*global_metrics[database_name][split]['mean_t_rr'][radius_rr])
                    if (log_false_positives and 'rr_failures' in global_metrics[database_name][split]
                        and len(global_metrics[database_name][split]['rr_failures']) > 0):
                        msg += '\n        Re-Ranking Failures (query, nn, nn_rerank, nn_dist, nn_rerank_dist):\n          '
                        msg += '\n          '.join([', '.join(x) for x in global_metrics[database_name][split]['rr_failures']])

                if len(local_metrics) == 0:
                    msg += '\n'
                    continue
                msg += '\n    Metric Localization:'
                for eval_mode in EVAL_MODES:
                    if eval_mode not in local_metrics[database_name][split]:
                        continue
                    msg += f'\n      Eval Mode: {eval_mode}'
                    for metric in local_metrics[database_name][split][eval_mode]:
                        if '_refined' in metric and not icp_refine:
                            continue
                        if 'failure_query_pos_ndx' in metric:
                            if not log_false_positives:
                                continue
                            msg += f"\n      {metric}: {local_metrics[database_name][split][eval_mode][metric]}"
                        else:
                            msg += f"\n      {metric}: {local_metrics[database_name][split][eval_mode][metric]:0.3f}"
                msg += '\n'
            msg += '\n'
        msg += "\n------------------------------------------------------------------------\n\n"
        f.write(msg)

def save_embeddings_to_file(global_embeddings, local_dict, model_name: str,
                            dataset_name: str, location_name: str, set_name: str = "database"):
    save_dir = os.path.join(os.path.dirname(__file__), "embeddings_v2", dataset_name, location_name, f"model_{model_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    global_embeddings_file = os.path.join(save_dir, f"{set_name}_global_embeddings.pickle")
    save_pickle(global_embeddings, global_embeddings_file)
    local_dict_file = os.path.join(save_dir, f"{set_name}_local_dict.pickle")
    save_pickle(local_dict, local_dict_file)

def load_embeddings_from_file(model_name: str, dataset_name: str, location_name: str, set_name: str = "database"):
    load_dir = os.path.join(os.path.dirname(__file__), "embeddings_v2", dataset_name, location_name, f"model_{model_name}")
    if not os.path.exists(load_dir):
        raise FileNotFoundError("No saved embeddings found for model. Run with --save_embeddings first.")
    else:
        global_embeddings_file = os.path.join(load_dir, f"{set_name}_global_embeddings.pickle")
        global_embeddings = load_pickle(global_embeddings_file)
        local_dict_file = os.path.join(load_dir, f"{set_name}_local_dict.pickle")
        local_dict = load_pickle(local_dict_file)
        assert (len(global_embeddings) * len(local_dict) > 0), (
            "Saved descriptors are corrupted, rerun with `save_descriptors` enabled"
        )
    return global_embeddings, local_dict

def cache_embeddings(global_embeddings, local_embeddings):
    """
    Saves embeddings to a tempfile for later use. 
    Files MUST be closed prior to exiting the program.
    """
    global_tmp_fp = tempfile.NamedTemporaryFile()
    local_tmp_fp = tempfile.NamedTemporaryFile()
    pickle.dump(global_embeddings, global_tmp_fp, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(local_embeddings, local_tmp_fp, protocol=pickle.HIGHEST_PROTOCOL)
    global_tmp_fp.seek(0)
    local_tmp_fp.seek(0)
    return global_tmp_fp, local_tmp_fp

def load_cached_embeddings(global_tmp_fp, local_tmp_fp):
    """
    Load cached embeddings from tempfile.
    """
    if global_tmp_fp is None:
        global_embeddings = None
    else:
        global_tmp_fp.seek(0)
        global_embeddings = pickle.load(global_tmp_fp)
        global_tmp_fp.seek(0)
    if local_tmp_fp is None:
        local_embeddings = None
    else:
        local_tmp_fp.seek(0)
        local_embeddings = pickle.load(local_tmp_fp)
        local_tmp_fp.seek(0)
    return global_embeddings, local_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PR and metric localisation with SGV reranking')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--radius', type=float, nargs='+', default=[5., 20.], help='True Positive thresholds in meters')
    parser.add_argument('--icp_refine', action='store_true', help='Refine estimated pose with ICP (unlike EgoNN, which refines GT pose [we handle this in dataloader])')
    parser.add_argument('--local_max_eval_threshold', type=float, default=np.inf,
                        help=('Maximum nn threshold to continue with local eval step '
                              'metric localisation not computed if distance to nearest retrieval is > thresh'))
    parser.add_argument('--num_neighbors', type=int, default=20, help='Number of nearest neighbours to consider in evaluation and re-ranking')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--log', action='store_true', help='Log false positives and top-5 retrievals')
    parser.add_argument('--only_global', action='store_true', help='Only run global (PR) evaluation')
    parser.add_argument('--use_ransac', action='store_true', help='Compare LGR with RANSAC in metric loc evaluation')
    parser.add_argument('--use_sgv', action='store_true', help='Use SGV for re-ranking')
    parser.add_argument('--sgv_d_thresh', type=float, default=0.4, help='Distance threshold used in SGV re-ranking')
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings to disk')
    parser.add_argument('--load_embeddings', action='store_true',
                        help=('Load embeddings from disk. Note this script will only check if '
                              'weights paths match, not if the configs used match.'))
    parser.add_argument('--print_false_positives', action='store_true', help='Print list of query and false positive retrieval indices')
    parser.add_argument('--unfair_rte_rre', action='store_true', help='Use unfair RTE and RRE evaluation from EgoNN (only computed on metric localisation successes)')
    parser.add_argument('--disable_reranking', action='store_true', help='Disable re-ranking evaluation to save time')
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print(f'Radius: {args.radius} [m]')
    print(f'Only global: {args.only_global}')
    print(f'ICP refine: {args.icp_refine}')
    print(f'Local max eval threshold: {args.local_max_eval_threshold}')
    print(f'Num neighbors: {args.num_neighbors}')
    print(f'Use RANSAC: {args.use_ransac}')
    print(f'Use SGV: {args.use_sgv}')
    print(f'SGV d thresh: {args.sgv_d_thresh}')
    if args.weights is None:
        if args.save_embeddings or args.load_embeddings:
            raise ValueError('Cannot save or load embeddings for random weights')
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    if args.save_embeddings and args.load_embeddings:
        print('[WARNING] Both save_embeddings AND load_embeddings specified, which is redundant. '
              'Will proceed without saving embeddings.')
        args.save_embeddings = False
    print(f'Save embeddings: {args.save_embeddings}')
    print(f'Load embeddings: {args.load_embeddings}')
    print(f'Print false positives: {args.print_false_positives}')
    print(f'Unfair RTE RRE: {args.unfair_rte_rre}')
    print(f'Disable re-ranking: {args.disable_reranking}')
    print('Debug mode: {}'.format(args.debug))
    print('Log search results: {}'.format(args.log))
    print('')

    set_seed()  # Seed RNG

    params = TrainingParams(
        args.config, args.model_config, debug=args.debug, verbose=args.verbose
    )
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    if args.weights is not None:
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
    else:
        model_name = w

    model.to(device)

    logging_level = 'DEBUG' if params.verbose else 'INFO'
    create_logger(log_file=None, logging_level=logging_level)

    # Check if metric localisation is supported by model
    if not params.local.enable_local and not args.only_global:
        msg = 'Metric localisation not supported by model... only running PR evaluation (pass `--only_global` to prevent this warning)'
        logging.warning(msg)
        args.only_global = True

    # Check if re-ranking
    reranking = False
    if not args.disable_reranking:
        if params.model_params.rerank_mode is not None or args.use_sgv:
            reranking = True
            if args.use_sgv:
                params.model_params.rerank_mode = 'sgv'

    # Override SGV d thresh
    params.sgv_d_thresh = args.sgv_d_thresh
    
    # Save results to the text file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    prefix = "Model Params: {}, Config: {}, Model: {}".format(model_params_name, config_name, model_name)

    global_metrics, local_metrics = evaluate(
        model,
        device,
        params,
        args.log,
        model_name,
        radius=args.radius,
        icp_refine=args.icp_refine,
        local_max_eval_threshold=args.local_max_eval_threshold,
        num_neighbors=args.num_neighbors,
        show_progress=True,
        only_global=args.only_global,
        use_ransac=args.use_ransac,
        save_embeddings=args.save_embeddings,
        load_embeddings=args.load_embeddings,
        reranking=reranking,
    )
    print_eval_stats(
        global_metrics,
        local_metrics,
        icp_refine=args.icp_refine,
        print_false_positives=args.print_false_positives,
        reranking=reranking,
        rerank_mode=params.model_params.rerank_mode,
    )

    write_eval_stats(
        f'metloc_rerank_{params.dataset_name}_split_results.txt',
        prefix,
        global_metrics,
        local_metrics,
        icp_refine=args.icp_refine,
        log_false_positives=args.print_false_positives,
        reranking=reranking,
        rerank_mode=params.model_params.rerank_mode,
    )
