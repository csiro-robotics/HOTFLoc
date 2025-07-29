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
import copy
from time import time
import tqdm
import ocnn
from typing import Sequence, List, Dict, Optional

from models.model_factory import model_factory
from misc.logger import create_logger
from misc.utils import TrainingParams, load_pickle, save_pickle
from misc.torch_utils import set_seed, to_device, release_cuda
from dataset.dataset_utils import make_eval_dataloader
from eval.utils import get_query_database_splits
from eval.sgv.sgv_utils import sgv_fn


def evaluate(
    model,
    device,
    params: TrainingParams,
    log: bool = False,
    model_name: str = "model",
    radius: Sequence[float] = [5., 20.],
    icp_refine: bool = False,
    local_max_eval_thresh: float = np.inf,
    num_neighbors: int = 20,
    show_progress: bool = False,
    save_embeddings: bool = False,
    load_embeddings: bool = False,
):
    # Run evaluation on all eval datasets
    eval_database_files, eval_query_files = get_query_database_splits(params)

    assert len(eval_database_files) == len(eval_query_files)
    stats = {}
    ave_recall = []
    ave_one_percent_recall = []
    ave_mrr = []
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
        temp = evaluate_dataset(
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
            local_max_eval_thresh=local_max_eval_thresh,
            num_neighbors=num_neighbors,
            show_progress=show_progress,
            save_embeddings=save_embeddings,
            load_embeddings=load_embeddings,
        )
        stats[location_name] = temp
        # 'average' key only exists when more than one split exists
        if 'average' in temp:
            ave_key = 'average'
        else:
            ave_key = next(iter(temp))
        ave_one_percent_recall.append(temp[ave_key]['ave_one_percent_recall'])
        ave_recall.append(temp[ave_key]['ave_recall'])
        ave_mrr.append(temp[ave_key]['ave_mrr'])
        
    # Compute average stats
    stats['average'] = {'average': {'ave_one_percent_recall': np.mean(ave_one_percent_recall),
                                    'ave_recall': np.mean(ave_recall, axis=0),
                                    'ave_mrr': np.mean(ave_mrr)}}
    return stats


def evaluate_dataset(
    model,
    device,
    params: TrainingParams,
    database_sets,
    query_sets,
    location_name: str,
    log: bool = False,
    model_name: str = "model",
    radius: Sequence[float] = [5., 20.],
    icp_refine: bool = False,
    local_max_eval_thresh: float = np.inf,
    num_neighbors: int = 20,
    show_progress: bool = False,
    save_embeddings: bool = False,
    load_embeddings: bool = False,
):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    stats = {}
    count = 0
    one_percent_recall = []
    mrr = []

    database_embeddings = []
    database_local_embeddings = []
    database_positions = []
    query_embeddings = []
    query_local_embeddings = []

    model.eval()

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
                temp_embeddings, temp_local_embeddings = get_latent_vectors(model, data_set, device, params, show_progress)
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
                temp_embeddings, temp_local_embeddings = get_latent_vectors(model, data_set, device, params, show_progress)
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
            else:
                split_name = os.path.split(os.path.split(query_sets[j][0]['query'])[0])[0]
            pair_recall, pair_opr, pair_mrr = get_metrics(
                m=i,
                n=j,
                database_global_embeddings=database_embeddings,
                query_global_embeddings=query_embeddings,
                database_local_embeddings=database_local_embeddings,
                query_local_embeddings=query_local_embeddings,
                database_positions=database_positions,
                query_sets=query_sets,
                database_sets=database_sets,
                radius=radius,
                icp_refine=icp_refine,
                local_max_eval_thresh=local_max_eval_thresh,
                num_neighbors=num_neighbors,
                log=log,
                model_name=model_name,
                show_progress=show_progress,
            )
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            mrr.append(pair_mrr)
            
            # Report per-split metrics
            stats[split_name] = {'ave_one_percent_recall': pair_opr,
                                 'ave_recall': pair_recall,
                                 'ave_mrr': pair_mrr}
    if count > 1:
        ave_recall = recall / count
        ave_one_percent_recall = np.mean(one_percent_recall)
        ave_mrr = np.mean(mrr)
        stats['average'] = {'ave_one_percent_recall': ave_one_percent_recall,
                            'ave_recall': ave_recall,
                            'ave_mrr': ave_mrr}
    del database_embeddings, database_local_embeddings, query_embeddings, query_local_embeddings
    return stats


def get_latent_vectors(model, data_set, device, params: TrainingParams,
                       show_progress: bool = False):
    # Adapted from original PointNetVLAD code
    if len(data_set) == 0:
        return None, None

    ### NOTE: Disabled so that eval can be tested during training debug mode
    # if params.debug:
    #     global_embeddings = np.random.rand(len(data_set), 256)
    #     local_embeddings = {4: torch.rand(len(data_set), 128, params.model_params.channels[1])}
    #     return global_embeddings, local_embeddings

    # Create dataloader for data_set
    dataloader = make_eval_dataloader(params, data_set, local=True)
    global_embeddings = None
    local_embeddings = {'coarse': [], 'fine': []}
    model.eval()
    with tqdm.tqdm(total=len(dataloader.dataset), disable=(not show_progress)) as pbar:
        for ii, batch in enumerate(dataloader):
            batch = to_device(batch, device, non_blocking=True)
            temp_global_embedding, temp_local_embedding = compute_embedding(model, batch)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(data_set), temp_global_embedding.shape[1]), dtype=temp_global_embedding.dtype)
            global_embeddings[ii*params.val_batch_size:(ii*params.val_batch_size + len(temp_global_embedding))] = temp_global_embedding
            # Split local embeddings from batch
            for embedding_resolution, embedding in temp_local_embedding.items():
                local_embeddings[embedding_resolution].extend(embedding)
            pbar.update(len(temp_global_embedding))
    
    return global_embeddings, local_embeddings


def compute_embedding(model, batch):
    with torch.inference_mode():
        # Compute global descriptor
        y = model(batch, global_only=True)
        global_embedding = release_cuda(y['global'], to_numpy=True)
        # Get local descriptors for each pyramid level
        local_embedding = None
        if 'local' in y:
            local_embedding = y['local']  # keep as tensors for future forward pass
            if 'octree' in y:
                # Only keep the coarse and fine indices to save mem
                local_depths = list(local_embedding.keys())
                depth_coarse = local_depths[model.coarse_idx]
                depth_fine = local_depths[model.fine_idx]
                # Batch stored in concat mode, so need to split back to batch elems
                batch_lengths_coarse = y['octree'].batch_nnum_nempty[depth_coarse].tolist()
                batch_lengths_fine = y['octree'].batch_nnum_nempty[depth_fine].tolist()
                local_embedding_coarse = release_cuda(local_embedding[depth_coarse].split(batch_lengths_coarse))
                local_embedding_fine = release_cuda(local_embedding[depth_fine].split(batch_lengths_fine))
                local_embedding = {'coarse': local_embedding_coarse, 'fine': local_embedding_fine}
            else:
                local_embedding = release_cuda(local_embedding)
        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return global_embedding, local_embedding


def get_metrics(
    m: int,
    n: int,
    database_global_embeddings: List[Optional[np.ndarray]],
    query_global_embeddings: List[Optional[np.ndarray]],
    database_local_embeddings: List[Optional[Dict[str, List[torch.Tensor]]]],
    query_local_embeddings: List[Optional[Dict[str, List[torch.Tensor]]]],
    database_positions: List[Optional[np.ndarray]],
    query_sets: List[Dict],
    database_sets: List[Dict],
    radius: Sequence[float] = [5., 20.],
    icp_refine: bool = False,
    local_max_eval_thresh: float = np.inf,
    num_neighbors: int = 20,
    log: bool = False,
    model_name: str = "model",
    show_progress: bool = False,
    only_global: bool = False,
):
    # eval_modes = ['Initial', 'Re-ranked']  # disabled until re-ranking implemented
    eval_modes = ['Initial']

    # ### TEMP FOR DEBUGGING ###
    # return np.ones(25, np.float32), 1.0, 1.0
    # ##########################

    # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
    global_metrics = {
        'tp': {r: [0] * num_neighbors for r in radius},
        'tp_rr': {r: [0] * num_neighbors for r in radius},
        'opr': {r: 0 for r in radius},
        'opr_rr': {r: 0 for r in radius},
        'RR': {r: [] for r in radius},
        'RR_rr': {r: [] for r in radius},
        't_RR': [],
        'rr_failures': [],
    }
    if only_global:
        local_metrics = {}
    else:
        local_metrics = {eval_mode: {'rre': [], 'rte': [], 'repeatability': [],
                                     'success': [], 'success_inliers': [], 'failure_inliers': [],
                                     'failure_query_pos_ndx': [], 'rre_refined': [],
                                     'rte_refined': [], 'success_refined': [],
                                     'success_inliers_refined': [], 'repeatability_refined': [],
                                     'failure_inliers_refined': [], 'failure_query_pos_ndx_refined': [],
                                     't_ransac': []} for eval_mode in eval_modes}

    # Original PointNetVLAD code
    database_global_output = database_global_embeddings[m]
    queries_global_output = query_global_embeddings[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_global_nbrs = KDTree(database_global_output)

    recall = [0] * num_neighbors
    recall_idx = []

    one_percent_retrieved = 0
    opr_threshold = max(int(round(len(database_global_output)/100.0)), 1)

    num_evaluated = 0
    for query_ndx in tqdm.tqdm(range(len(queries_global_output)), disable=(not show_progress)):
        query_details = query_sets[n][query_ndx]    # {'query': path, 'northing': , 'easting': }
        query_position = np.array((query_details['northing'], query_details['easting']))
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, nn_indices = database_global_nbrs.query(queries_global_output[query_ndx][None,:], k=num_neighbors)

        # Euclidean distance between the query and nn
        delta = query_position - database_positions[m][nn_indices.squeeze(axis=0)]       # (k, 2) array
        euclid_dist = np.linalg.norm(delta, axis=-1)     # (k,) array

        if log:
            # Log false positives (returned as the first element)
            # Check if there's a false positive returned as the first element
            if nn_indices[0][0] not in true_neighbors:
                fp_ndx = nn_indices[0][0]
                fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0, 0]  # Distance in embedding space
                fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
                                        (query_details['easting'] - fp['easting']) ** 2)
                # Find the first true positive
                tp = None
                for k in range(len(nn_indices[0])):
                    if nn_indices[0][k] in true_neighbors:
                        closest_pos_ndx = nn_indices[0][k]
                        tp = database_sets[m][closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                        tp_emb_dist = distances[0][k]
                        tp_world_dist = np.sqrt((query_details['northing'] - tp['northing']) ** 2 +
                                                (query_details['easting'] - tp['easting']) ** 2)
                        break
                            
                out_fp_file_name = f"{model_name}_log_fp.txt"
                with open(out_fp_file_name, "a") as f:
                    s = "{}, {}, {:0.2f}, {:0.2f}".format(query_details['query'], fp['query'], fp_emb_dist, fp_world_dist)
                    if tp is None:
                        s += ', 0, 0, 0\n'
                    else:
                        s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
                    f.write(s)

            # Save details of 5 best matches for later visualization for 1% of queries
            s = f"{query_details['query']}, {query_details['northing']}, {query_details['easting']}"
            for k in range(min(len(nn_indices[0]), 5)):
                is_match = nn_indices[0][k] in true_neighbors
                e_ndx = nn_indices[0][k]
                e = database_sets[m][e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
                e_emb_dist = distances[0][k]
                world_dist = np.sqrt((query_details['northing'] - e['northing']) ** 2 +
                                        (query_details['easting'] - e['easting']) ** 2)
                s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
            s += '\n'
            out_top5_file_name = f"{model_name}_log_search_results.txt"
            with open(out_top5_file_name, "a") as f:
                f.write(s)

        # # Re-Ranking with SGV
        # # NOTE: TO DO THIS, NEED TO PRE-COMPUTE THE COARSE CENTROIDS, OR GET THEM AFTER RUNNING FORWARD PASS OF HOTFORMERMETRICLOC
        # topk = min(num_neighbors, len(nn_indices))
        # tick = time()
        # candidate_local_embeddings = database_local_embeddings[m][nn_indices]
        # candidate_keypoints = local_map_embeddings_keypoints[m][nn_indices]
        # fitness_list = sgv_fn(query_local_embeddings[n][query_ndx], candidate_local_embeddings, candidate_keypoints, d_thresh=0.4)
        # topk_rerank = np.flip(np.asarray(fitness_list).argsort())
        # topk_rerank_inds = copy.deepcopy(nn_indices)
        # topk_rerank_inds[:topk] = nn_indices[topk_rerank]
        # t_rerank = time() - tick
        # global_metrics['t_RR'].append(t_rerank)

        # delta_rerank = query_position - database_positions[m][topk_rerank_inds]
        # euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)

        # # Log cases where re-ranking is worse (causes PR failure, or is
        # #   significantly worse than the original top-candidate)
        # rr_to_nn_euclid_dist = euclid_dist_rr[0] - euclid_dist[0]
        # if (euclid_dist[0] <= self.MAX_NN_EUCLID_DIST
        #         and rr_to_nn_euclid_dist > self.MAX_RR_TO_NN_EUCLID_DIST):
        #     # print(f'Fail: {euclid_dist_rr[0]:.2f}m > {euclid_dist[0]:.2f}m', flush=True)
        #     query_name = os.path.basename(self.eval_set.query_set[query_ndx].rel_scan_filepath)
        #     nn_name = os.path.basename(self.eval_set.map_set[nn_indices[0]].rel_scan_filepath)
        #     nn_rerank_name = os.path.basename(self.eval_set.map_set[topk_rerank_inds[0]].rel_scan_filepath)
        #     global_metrics['rr_failures'].append((query_name, nn_name,
        #                                             nn_rerank_name,
        #                                             f'{euclid_dist[0]:.2f}',
        #                                             f'{euclid_dist_rr[0]:.2f}'))

        # OLD METRICS:
        # for j in range(len(nn_indices[0])):
        #     if nn_indices[0][j] in true_neighbors:
        #         recall[j] += 1
        #         recall_idx.append(j+1)
        #         break

        # if len(list(set(nn_indices[0][0:opr_threshold]).intersection(set(true_neighbors)))) > 0:
        #     one_percent_retrieved += 1

        # Count true positives and 1% retrieved for different radius and NN number
        global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in radius}
        # global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in self.radius}
        global_metrics['opr'] = {r: global_metrics['opr'][r] + (1 if (euclid_dist[:opr_threshold] <= r).any() else 0) for r in radius}
        # global_metrics['opr_rr'] = {r: global_metrics['opr_rr'][r] + (1 if (euclid_dist_rr[:threshold] <= r).any() else 0) for r in self.radius}
        global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in radius}
        # global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}
        if only_global:
            continue

        # TODO: Metric Localisation bit


    # OLD METRICS:
    # one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    # recall = (np.cumsum(recall)/float(num_evaluated))*100
    # mrr = np.mean(1/np.array(recall_idx))*100

    # Calculate mean metrics
    global_metrics['recall'] = {r: [global_metrics['tp'][r][nn] / num_evaluated for nn in range(num_neighbors)] for r in radius}
    # global_metrics['recall_rr'] = {r: [global_metrics['tp_rr'][r][nn] / num_evaluated for nn in range(num_neighbors)] for r in radius}
    global_metrics['recall@1'] = {r: global_metrics['recall'][r][0] for r in radius}
    # global_metrics['recall@1_rr'] = {r: global_metrics['recall_rr'][r][0] for r in radius}
    global_metrics['recall@1%'] = {r: global_metrics['opr'][r] / num_evaluated for r in radius}
    # global_metrics['recall@1%_rr'] = {r: global_metrics['opr_rr'][r] / num_evaluated for r in radius}
    global_metrics['MRR'] = {r: np.mean(np.asarray(global_metrics['RR'][r])) for r in radius}
    # global_metrics['MRR_rr'] = {r: np.mean(np.asarray(global_metrics['RR_rr'][r])) for r in radius}
    # global_metrics['mean_t_RR'] = np.mean(np.asarray(global_metrics['t_RR']))

    mean_local_metrics = {}
    if not only_global:
        # Calculate mean values of local descriptor metrics
        for eval_mode in eval_modes:
            mean_local_metrics[eval_mode] = {}
            for metric in local_metrics[eval_mode]:
                m_l = local_metrics[eval_mode][metric]
                if len(m_l) == 0:
                    mean_local_metrics[eval_mode][metric] = 0.
                else:
                    if 'failure_query_pos_ndx' in metric:  # we want a list of all query + pos failure pairs
                        mean_local_metrics[eval_mode][metric] = m_l
                        continue
                    if metric == 't_ransac':
                        mean_local_metrics[eval_mode]['t_ransac_sd'] = np.std(m_l)
                    mean_local_metrics[eval_mode][metric] = np.mean(m_l)

    return global_metrics, local_metrics

def print_eval_stats(stats):
    # Altered to expect per-split stats
    msg = 'Eval Results'
    for database_name in stats:
        msg += '\nDataset: {}\n'.format(database_name)
        for split in stats[database_name]:
            msg += '    Split: {}\n'.format(split)
            t = '    Avg. top 1% recall: {:.2f}   Avg. MRR: {:.2f}   Avg. recall @N:\n'
            msg += t.format(
                stats[database_name][split]["ave_one_percent_recall"],
                stats[database_name][split]["ave_mrr"],
            )
            msg += '    ' + str(stats[database_name][split]['ave_recall']).replace('\n','\n    ')
    logging.info(msg)


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    # ave_1p_recall_l = []
    # ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            s += f"\n[{ds}]\n"
            for split in stats[ds]:
                s += f'    Split: [{split}]\n'
                ave_1p_recall = stats[ds][split]['ave_one_percent_recall']
                # ave_1p_recall_l.append(ave_1p_recall)
                ave_recall = stats[ds][split]['ave_recall'][0]
                # ave_recall_l.append(ave_recall)
                ave_mrr = stats[ds][split]['ave_mrr']
                s += '    AR@1%: {:0.2f}, AR@1: {:0.2f}, MRR: {:0.2f}, AR@N:\n'.format(ave_1p_recall, ave_recall, ave_mrr)
                s += '    ' + str(stats[ds][split]['ave_recall']).replace('\n','\n    ')
                s += '\n'

        # NOTE: below is redundant as average is now stored in stats dict
        # mean_1p_recall = np.mean(ave_1p_recall_l)
        # mean_recall = np.mean(ave_recall_l)
        # s += "\n\nMean AR@1%: {:0.2f}, Mean AR@1: {:0.2f}\n\n".format(mean_1p_recall, mean_recall)
        s += "\n------------------------------------------------------------------------\n\n"
        f.write(s)


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
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--radius', type=float, nargs='+', default=[5., 20.], help='True Positive thresholds in meters')
    parser.add_argument('--icp_refine', action='store_true', help='Refine estimated pose with ICP')
    parser.add_argument('--local_max_eval_thresh', type=float, default=np.inf,
                        help=('Maximum nn threshold to continue with local eval step '
                              'metric localisation not computed if distance to nearest retrieval is > thresh'))
    parser.add_argument('--num_neighbors', type=int, default=20, help='Number of nearest neighbours to consider in evaluation and re-ranking')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--log', action='store_true', help='Log false positives and top-5 retrievals')
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings to disk')
    parser.add_argument('--load_embeddings', action='store_true',
                        help=('Load embeddings from disk. Note this script will only check if '
                              'weights paths match, not if the configs used match.'))
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print(f'Radius: {args.radius} [m]')
    print(f'ICP refine: {args.icp_refine}')
    print(f'Local max eval threshold: {args.local_max_eval_thresh}')
    print(f'Num neighbors: {args.num_neighbors}')
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

    model = model_factory(params.model_params)
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
    
    # Save results to the text file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    prefix = "Model Params: {}, Config: {}, Model: {}".format(model_params_name, config_name, model_name)

    stats = evaluate(
        model,
        device,
        params,
        args.log,
        model_name,
        radius=args.radius,
        icp_refine=args.icp_refine,
        local_max_eval_thresh=args.local_max_eval_thresh,
        num_neighbors=args.num_neighbors,
        show_progress=True,
        save_embeddings=args.save_embeddings,
        load_embeddings=args.load_embeddings,
    )
    print_eval_stats(stats)

    pnv_write_eval_stats(f"pnv_{params.dataset_name}_split_results.txt", prefix, stats)
