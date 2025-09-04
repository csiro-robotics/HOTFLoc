# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
# Adapted to process samples in batches by Ethan Griffiths (QUT & CSIRO Data61).
# 24-05-2025 Further adapted to implement SpectralGV re-ranking: https://github.com/csiro-robotics/SpectralGV/blob/main
# 29-07-2025 Adapted to include metric localisation evaluation

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import logging
import torch
import time
import tqdm
from typing import Sequence, List, Dict, Optional

from dataset.dataset_utils import make_eval_dataloader, make_eval_dataloader_6DOF
from dataset.augmentation import Normalize
from eval.utils import get_query_database_splits
from eval.sgv.sgv_utils import sgv_fn
from eval.geotransformer_utils import compute_geotransformer_metrics
from misc.logger import create_logger
from misc.point_clouds import icp
from misc.torch_utils import set_seed, to_device, release_cuda
from misc.utils import TrainingParams, load_pickle, save_pickle
from models.model_factory import model_factory
from models.hotformerloc_metric_loc import HOTFormerMetricLoc

DISABLE_ICP = False
# EVAL_MODES = ['Initial', 'Re-Ranked']
EVAL_MODES = ['Initial']  # re-ranking temporarily disabled for debugging


def evaluate(
    model: torch.nn.Module,
    device,
    params: TrainingParams,
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
):
    # Run evaluation on all eval datasets
    eval_database_files, eval_query_files = get_query_database_splits(params)

    assert len(eval_database_files) == len(eval_query_files)
    global_metrics, local_metrics = {}, {}
    average_global_metrics, average_local_metrics = {}, {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
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
):
    # Run evaluation on a single dataset
    global_metrics, local_metrics = {}, {}

    database_embeddings = []
    database_local_embeddings = []
    database_positions = []
    query_embeddings = []
    query_local_embeddings = []

    model.eval()

    # TODO: Determine why memory usage peaks at ~200GB when computing/loading Venman from disk (even though Kara is larger)
    if show_progress:
        logging.info(f'{"Loading" if load_embeddings else "Computing"} database embeddings')
    for ii, data_set in enumerate(database_sets):
        temp_embeddings, temp_local_embeddings, temp_positions = [None]*3
        if len(data_set) > 0:
            # Create array of coordinates of all db elements
            temp_positions = np.array([(db_details['northing'], db_details['easting']) for db_details in data_set.values()])
            if load_embeddings:
                temp_embeddings, temp_local_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'database_{ii}')
            else:
                temp_embeddings, temp_local_embeddings = get_latent_vectors(model, data_set, device, params, only_global, show_progress)
            if save_embeddings:
                save_embeddings_to_file(temp_embeddings, temp_local_embeddings, model_name, params.dataset_name, location_name, f'database_{ii}')
        database_embeddings.append(temp_embeddings)
        database_local_embeddings.append(temp_local_embeddings)
        database_positions.append(temp_positions)

    if show_progress:
        logging.info(f'{"Loading" if load_embeddings else "Computing"} query embeddings')
    for jj, data_set in enumerate(query_sets):
        temp_embeddings, temp_local_embeddings = [None]*2
        if len(data_set) > 0:
            if load_embeddings:
                temp_embeddings, temp_local_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'query_{jj}')
            else:
                temp_embeddings, temp_local_embeddings = get_latent_vectors(model, data_set, device, params, only_global, show_progress)
            if save_embeddings:
                save_embeddings_to_file(temp_embeddings, temp_local_embeddings, model_name, params.dataset_name, location_name, f'query_{jj}')
        query_embeddings.append(temp_embeddings)
        query_local_embeddings.append(temp_local_embeddings)

    if show_progress:
        logging.info('Running evaluation')
    for i in range(len(database_sets)):
        for j in range(len(query_sets)):
            if (i == j and params.skip_same_run) or database_embeddings[i] is None or query_embeddings[j] is None:
                continue
            if 'CSCampus3D' in params.dataset_name:
                # For Campus3D, we report on the aerial-only database, which is idx 1
                if i != 1:
                    continue
                split_name = os.path.split(os.path.split(database_sets[i][0]['query'])[0])[0] + f'_idx{i}'
            elif params.dataset_name == 'WildPlaces':
                # For WildPlaces, there are multiple databases per query set, so add both to split name
                split_name = (os.path.split(os.path.split(database_sets[i][0]['query'])[0])[0]
                              + '-' + query_sets[j][0]['query'].split('/')[1])
            else:
                split_name = os.path.split(os.path.split(query_sets[j][0]['query'])[0])[0]
            temp_global_metrics, temp_local_metrics = get_metrics(
                m=i,
                n=j,
                database_global_embeddings=database_embeddings[i],
                query_global_embeddings=query_embeddings[j],
                database_local_embeddings=database_local_embeddings[i],
                query_local_embeddings=query_local_embeddings[j],
                database_positions=database_positions[i],
                database_set=database_sets[i],
                query_set=query_sets[j],
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
            )
            # Report per-split metrics
            global_metrics[split_name] = temp_global_metrics
            if not only_global:
                local_metrics[split_name] = temp_local_metrics

    # Compute average for split
    global_metrics['average'] = average_nested_dict(global_metrics)
    if not only_global:
        local_metrics['average'] = average_nested_dict(local_metrics)

    del database_embeddings, database_local_embeddings, query_embeddings, query_local_embeddings
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

def get_latent_vectors(model: torch.nn.Module, data_set: Dict, device,
                       params: TrainingParams, only_global: bool = False,
                       show_progress: bool = False):
    # Adapted from original PointNetVLAD code
    if len(data_set) == 0:
        return None, None

    ### NOTE: Disabled so that eval can be tested during training debug mode
    if params.debug:
        global_embeddings = np.random.randn(len(data_set), params.model_params.output_dim)
        local_embeddings = {'coarse': [], 'fine': []}
        if not only_global:
            for _ in range(len(data_set)):
                local_embeddings['coarse'].append(torch.randn(128, params.model_params.channels[-1]))
                local_embeddings['fine'].append(torch.randn(512, params.model_params.channels[-3]))
        return global_embeddings, local_embeddings

    # Create dataloader for data_set
    dataloader = make_eval_dataloader(params, data_set)
    global_embeddings = None
    local_embeddings = {'coarse': [], 'fine': []}
    model.eval()
    with tqdm.tqdm(total=len(dataloader.dataset), disable=(not show_progress)) as pbar:
        for ii, batch_dict in enumerate(dataloader):
            batch = batch_dict['batch']
            batch = to_device(batch, device, non_blocking=True, construct_octree_neigh=True)
            temp_global_embedding, temp_local_embedding = compute_embedding(model, batch, only_global)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(data_set), temp_global_embedding.shape[1]), dtype=temp_global_embedding.dtype)
            global_embeddings[ii*params.val_batch_size:(ii*params.val_batch_size + len(temp_global_embedding))] = temp_global_embedding
            # Split local embeddings from batch
            if temp_local_embedding is not None:
                for embedding_resolution, embedding in temp_local_embedding.items():
                    local_embeddings[embedding_resolution].extend(embedding)
            pbar.update(len(temp_global_embedding))
    
    return global_embeddings, local_embeddings


def compute_embedding(model: torch.nn.Module, batch: Dict, only_global: bool = False):
    with torch.inference_mode():
        # Compute global descriptor
        y = model(batch, global_only=True)
        global_embedding = release_cuda(y['global'], to_numpy=True)
        # Get local descriptors for each pyramid level
        local_embedding = None
        if not only_global and 'local' in y:
            local_embedding = y['local']  # keep as tensors for future forward pass
            if isinstance(model, HOTFormerMetricLoc):
                # Only keep the coarse and fine indices to save mem
                # Batch stored in concat mode, so need to split back to batch elems
                batch_lengths_coarse = y['octree'].batch_nnum_nempty[model.depth_coarse].tolist()
                batch_lengths_fine = y['octree'].batch_nnum_nempty[model.depth_fine].tolist()
                local_embedding_coarse = release_cuda(local_embedding[model.depth_coarse].split(batch_lengths_coarse))
                local_embedding_fine = release_cuda(local_embedding[model.depth_fine].split(batch_lengths_fine))
                local_embedding = {'coarse': local_embedding_coarse, 'fine': local_embedding_fine}
            else:
                local_embedding = release_cuda(local_embedding)
        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return global_embedding, local_embedding


def get_metrics(
    m: int,
    n: int,
    database_global_embeddings: np.ndarray,
    query_global_embeddings: np.ndarray,
    database_local_embeddings: Dict[str, List[torch.Tensor]],
    query_local_embeddings: Dict[str, List[torch.Tensor]],
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
    }
    intermediate_metrics = {
        'tp': {r: [0] * num_neighbors for r in radius},
        'tp_rr': {r: [0] * num_neighbors for r in radius},
        'opr': {r: 0 for r in radius},
        'opr_rr': {r: 0 for r in radius},
        'RR': {r: [] for r in radius},
        'RR_rr': {r: [] for r in radius},
        't_rr': [],
    }
    local_metrics = {}
    if not only_global:
        for eval_mode in EVAL_MODES:
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
        # # Re-Ranking with SGV
        # # NOTE: TO DO THIS, NEED TO PRE-COMPUTE THE COARSE CENTROIDS, OR GET THEM AFTER RUNNING FORWARD PASS OF HOTFORMERMETRICLOC
        # topk = min(num_neighbors, len(nn_indices))
        # tick = time.perf_counter()
        # candidate_local_embeddings = database_local_embeddings[m][nn_indices]
        # candidate_keypoints = local_map_embeddings_keypoints[m][nn_indices]
        # fitness_list = sgv_fn(query_local_embeddings[n][query_idx], candidate_local_embeddings, candidate_keypoints, d_thresh=0.4)
        # topk_rerank = np.flip(np.asarray(fitness_list).argsort())
        # topk_rerank_indices = copy.deepcopy(nn_indices)
        # topk_rerank_indices[:topk] = nn_indices[topk_rerank]
        # t_rerank = time.perf_counter() - tick
        # intermediate_metrics['t_rr'].append(t_rerank)

        # delta_rerank = query_position - database_positions[m][topk_rerank_indices]
        # euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)

        # # Log cases where re-ranking is worse (causes PR failure, or is
        # #   significantly worse than the original top-candidate)
        # rr_to_nn_euclid_dist = euclid_dist_rr - euclid_dist
        # if (euclid_dist <= self.MAX_NN_EUCLID_DIST
        #         and rr_to_nn_euclid_dist > self.MAX_RR_TO_NN_EUCLID_DIST):
        #     # print(f'Fail: {euclid_dist_rr:.2f}m > {euclid_dist:.2f}m', flush=True)
        #     query_name = os.path.basename(self.eval_set.query_set[query_idx].rel_scan_filepath)
        #     nn_name = os.path.basename(self.eval_set.map_set[nn_indices].rel_scan_filepath)
        #     nn_rerank_name = os.path.basename(self.eval_set.map_set[topk_rerank_indices].rel_scan_filepath)
        #     global_metrics['rr_failures'].append((query_name, nn_name,
        #                                             nn_rerank_name,
        #                                             f'{euclid_dist:.2f}',
        #                                             f'{euclid_dist_rr:.2f}'))
        ########################################################################

        # Count true positives and 1% retrieved for different radius and NN number
        intermediate_metrics['tp'] = {r: [intermediate_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in radius}
        # intermediate_metrics['tp_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in self.radius}
        intermediate_metrics['opr'] = {r: intermediate_metrics['opr'][r] + (1 if (euclid_dist[:opr_threshold] <= r).any() else 0) for r in radius}
        # intermediate_metrics['opr_rr'] = {r: intermediate_metrics['opr_rr'][r] + (1 if (euclid_dist_rr[:threshold] <= r).any() else 0) for r in self.radius}
        intermediate_metrics['RR'] = {r: intermediate_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in radius}
        # intermediate_metrics['RR_rr'] = {r: intermediate_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}
        if only_global:
            continue

        # LOCAL DESCRIPTOR EVALUATION
        # Do the evaluation only if the nn pose is within distance threshold
        # Otherwise the overlap is too small to get reasonable results
        # (evaluation continues if standard OR re-ranked nn pose is within thresh meters)
        if euclid_dist[0] > local_max_eval_threshold:  # and euclid_dist_rr[0] > local_max_eval_threshold:
            continue

        # Cache query and nn idx for metric loc eval
        metric_loc_pairs_list['Initial'].append((query_idx, nn_indices[0]))
        # pairs_dict['Re-Ranked'].append(query_idx, topk_rerank_indices[0]))  # TODO: Enable when re-ranking implemented

    # Run metric localisation evaluation for initial and re-ranked top-candidate
    for eval_mode in EVAL_MODES:
        if only_global:
            break
        eval_dataloader = make_eval_dataloader_6DOF(
            params, query_set, database_set, metric_loc_pairs_list[eval_mode]
        )
        # TODO: Process with batch size > 1 to reduce dataloader bottleneck
        for idx, batch in tqdm.tqdm(enumerate(eval_dataloader),
                                    total=len(eval_dataloader),
                                    desc=f'Metric Localisation [{eval_mode}]',
                                    disable=(not show_progress)):
            query_idx, nn_idx = metric_loc_pairs_list[eval_mode][idx]
            T_gt = batch['transform'][0]

            # Use pre-computed embeddings so we don't re-compute the entire forward pass
            if params.debug:
                if idx >= 2:
                    break
                batch['anc_local_feats'] = {
                    'coarse': torch.randn(batch['anc_batch']['octree'].batch_nnum_nempty[model.depth_coarse], params.model_params.channels[-1]),
                    'fine': torch.randn(batch['anc_batch']['octree'].batch_nnum_nempty[model.depth_fine], params.model_params.channels[-3]),
                }
                batch['pos_local_feats'] = {
                    'coarse': torch.randn(batch['pos_batch']['octree'].batch_nnum_nempty[model.depth_coarse], params.model_params.channels[-1]),
                    'fine': torch.randn(batch['pos_batch']['octree'].batch_nnum_nempty[model.depth_fine], params.model_params.channels[-3]),
                }
            else:
                batch['anc_local_feats'] = {
                    'coarse': query_local_embeddings['coarse'][query_idx],
                    'fine': query_local_embeddings['fine'][query_idx],
                }
                batch['pos_local_feats'] = {
                    'coarse': database_local_embeddings['coarse'][nn_idx],
                    'fine': database_local_embeddings['fine'][nn_idx],
                }
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

            # TODO: Re-add support for non-geotransformer variants

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

    # TODO: ADD METRICS TO TRAINER.py WANDB
    # Calculate mean global metrics
    global_metrics['recall'] = {r: [intermediate_metrics['tp'][r][nn] / num_evaluated for nn in range(num_neighbors)] for r in radius}
    # global_metrics['recall_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] / num_evaluated for nn in range(num_neighbors)] for r in radius}
    global_metrics['recall@1'] = {r: global_metrics['recall'][r][0] for r in radius}
    # global_metrics['recall@1_rr'] = {r: global_metrics['recall_rr'][r][0] for r in radius}
    global_metrics['recall@1%'] = {r: intermediate_metrics['opr'][r] / num_evaluated for r in radius}
    # global_metrics['recall@1%_rr'] = {r: intermediate_metrics['opr_rr'][r] / num_evaluated for r in radius}
    global_metrics['MRR'] = {r: np.mean(np.asarray(intermediate_metrics['RR'][r])) for r in radius}
    # global_metrics['MRR_rr'] = {r: np.mean(np.asarray(intermediate_metrics['RR_rr'][r])) for r in radius}
    # global_metrics['mean_t_rr'] = np.mean(np.asarray(intermediate_metrics['t_rr']))  # NOTE: this will break average_nested_dict(), so fix it

    mean_local_metrics = {}
    if not only_global:
        # Calculate mean values of local descriptor metrics
        for eval_mode in EVAL_MODES:
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

def print_eval_stats(global_metrics: Dict, local_metrics: Dict, icp_refine=False, print_false_positives=False):
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

            if 'Re-Ranked' in EVAL_MODES:
                msg += '\n    Re-Ranking:'
                recall_rr = global_metrics[database_name][split]['recall_rr']
                for radius_rr in recall_rr.keys():
                    msg += f"\n      Radius: {radius_rr} [m]:\n        Recall@N: "
                    for ii, x in enumerate(recall_rr[radius_rr]):
                        msg += f"{x:0.3f}, "
                        if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                            msg += "\n                  "
                    msg += "\n        Recall@1%: {:0.3f}".format(global_metrics[database_name][split]['recall@1%_rr'][radius_rr])
                    msg += '\n        MRR: {:0.3f}'.format(global_metrics[database_name][split]['MRR_rr'][radius_rr])
                msg += '\n        Re-Ranking Time: {:0.3f} [ms]'.format(1000.0 *global_metrics[database_name][split]['mean_t_rr'])
                if print_false_positives and 'rr_failures' in global_metrics[database_name][split]:
                    msg += '\n        Re-Ranking Failures (query, nn, nn_rerank, nn_dist, nn_rerank_dist):\n          '
                    msg += '\n          '.join(global_metrics[database_name][split]['rr_failures'])

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

                if 'Re-Ranked' in EVAL_MODES:
                    msg += '\n    Re-Ranking:'
                    recall_rr = global_metrics[database_name][split]['recall_rr']
                    for radius_rr in recall_rr.keys():
                        msg += f"\n      Radius: {radius_rr} [m]:\n        Recall@N: "
                        for ii, x in enumerate(recall_rr[radius_rr]):
                            msg += f"{x:0.3f}, "
                            if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                                msg += "\n                  "
                        msg += "\n        Recall@1%: {:0.3f}".format(global_metrics[database_name][split]['recall@1%_rr'][radius_rr])
                        msg += '\n        MRR: {:0.3f}'.format(global_metrics[database_name][split]['MRR_rr'][radius_rr])
                    msg += '\n        Re-Ranking Time: {:0.3f} [ms]'.format(1000.0 *global_metrics[database_name][split]['mean_t_rr'])
                    if log_false_positives and 'rr_failures' in global_metrics[database_name][split]:
                        msg += '\n        Re-Ranking Failures (query, nn, nn_rerank, nn_dist, nn_rerank_dist):\n          '
                        msg += '\n          '.join(global_metrics[database_name][split]['rr_failures'])

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


def save_embeddings_to_file(global_embeddings, local_embeddings, model_name: str,
                            dataset_name: str, location_name: str, set_name: str = "database"):
    save_dir = os.path.join(os.path.dirname(__file__), "embeddings", dataset_name, location_name, f"model_{model_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    global_embeddings_file = os.path.join(save_dir, f"{set_name}_global_embeddings.pickle")
    save_pickle(global_embeddings, global_embeddings_file)
    local_embeddings_file = os.path.join(save_dir, f"{set_name}_local_embeddings.pickle")
    save_pickle(local_embeddings, local_embeddings_file)


def load_embeddings_from_file(model_name: str, dataset_name: str, location_name: str, set_name: str = "database"):
    load_dir = os.path.join(os.path.dirname(__file__), "embeddings", dataset_name, location_name, f"model_{model_name}")
    if not os.path.exists(load_dir):
        raise FileNotFoundError("No saved embeddings found for model. Run with --save_embeddings first.")
    else:
        global_embeddings_file = os.path.join(load_dir, f"{set_name}_global_embeddings.pickle")
        global_embeddings = load_pickle(global_embeddings_file)
        local_embeddings_file = os.path.join(load_dir, f"{set_name}_local_embeddings.pickle")
        local_embeddings = load_pickle(local_embeddings_file)
        assert (len(global_embeddings) * len(local_embeddings) > 0), (
            "Saved descriptors are corrupted, rerun with `save_descriptors` enabled"
        )
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
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--log', action='store_true', help='Log false positives and top-5 retrievals')
    parser.add_argument('--only_global', action='store_true', help='Only run global (PR) evaluation')
    parser.add_argument('--use_ransac', action='store_true', help='Compare LGR with RANSAC in metric loc evaluation')
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings to disk')
    parser.add_argument('--load_embeddings', action='store_true',
                        help=('Load embeddings from disk. Note this script will only check if '
                              'weights paths match, not if the configs used match.'))
    parser.add_argument('--print_false_positives', action='store_true', help='Print list of query and false positive retrieval indices')
    parser.add_argument('--unfair_rte_rre', action='store_true', help='Use unfair RTE and RRE evaluation from EgoNN (only computed on metric localisation successes)')
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print(f'Radius: {args.radius} [m]')
    print(f'Only global: {args.only_global}')
    print(f'ICP refine: {args.icp_refine}')
    print(f'Local max eval threshold: {args.local_max_eval_threshold}')
    print(f'Num neighbors: {args.num_neighbors}')
    print(f'Use RANSAC: {args.use_ransac}')
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
    print('Debug mode: {}'.format(args.debug))
    print('Log search results: {}'.format(args.log))
    print('')

    set_seed()  # Seed RNG

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
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
        if os.path.splitext(args.weights)[1] == '.ckpt':
            state = torch.load(args.weights)
            model.load_state_dict(state['model_state_dict'])
        else:  # .pt or .pth
            model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        model_name = w

    model.to(device)

    logging_level = 'DEBUG' if params.debug else 'INFO'
    create_logger(log_file=None, logging_level=logging_level)

    # Check if metric localisation is supported by model
    if not params.local.enable_local and not args.only_global:
        msg = 'Metric localisation not supported by model... only running PR evaluation (pass `--only_global` to prevent this warning)'
        logging.warning(msg)
        args.only_global = True
    
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
    )
    print_eval_stats(
        global_metrics, local_metrics, icp_refine=args.icp_refine, print_false_positives=args.print_false_positives,
    )

    write_eval_stats(
        f"metloc_sgv_{params.dataset_name}_split_results.txt",
        prefix,
        global_metrics,
        local_metrics,
        icp_refine=args.icp_refine,
        log_false_positives=args.print_false_positives,
    )
