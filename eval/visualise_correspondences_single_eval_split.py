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

from dataset.dataset_utils import make_eval_dataloader_6DOF
from eval.vis_utils import (
    colourise_points_by_similarity,
    visualise_coarse_correspondences,
    visualise_fine_correspondences,
    visualise_registration,
    visualise_LGR_initial_registration,
    visualise_similarity,
)
from eval.evaluate_metric_loc_splits_sgv import load_embeddings_from_file, save_embeddings_to_file, get_latent_vectors
from eval.utils import get_query_database_splits
from models.model_factory import model_factory
from models.losses.geotransformer_loss import Evaluator
from misc.torch_utils import set_seed, release_cuda, to_device
from misc.utils import TrainingParams
from misc.logger import create_logger
from geotransformer.utils.visualization import (
    draw_point_to_node,
    draw_node_correspondences,
    get_colors_with_tsne
)

# EVAL_MODES = ['Initial']  # re-ranking temporarily disabled
EVAL_MODE = 'Initial'


def get_eval_pairs_dataloader(
    params: TrainingParams,
    model: torch.nn.Module,
    model_name: str,
    device: str,
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
        database_embeddings, database_local_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'database_{args.database_split_idx}')
    else:
        database_embeddings, database_local_embeddings = get_latent_vectors(model, database_set, device, params, only_global=False, show_progress=True)
    if args.save_embeddings:
        save_embeddings_to_file(database_embeddings, database_local_embeddings, model_name, params.dataset_name, location_name, f'database_{args.database_split_idx}')

    if args.load_embeddings:
        query_embeddings, query_local_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'query_{args.query_split_idx}')
    else:
        query_embeddings, query_local_embeddings = get_latent_vectors(model, query_set, device, params, only_global=False, show_progress=True)
    if args.save_embeddings:
        save_embeddings_to_file(query_embeddings, query_local_embeddings, model_name, params.dataset_name, location_name, f'query_{args.query_split_idx}')

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

    metric_loc_pairs_list = {EVAL_MODE: []}

    opr_threshold = max(int(round(len(database_embeddings)/100.0)), 1)

    num_evaluated = 0
    for query_idx in tqdm.tqdm(range(len(query_embeddings)),
                               desc='Place Recognition',):
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

        ########################################################################
        # # Re-Ranking with SGV
        # # NOTE: TO DO THIS, NEED TO PRE-COMPUTE THE COARSE CENTROIDS, OR GET THEM AFTER RUNNING FORWARD PASS OF HOTFORMERMETRICLOC
        # topk = min(num_neighbors, len(nn_indices))
        # tick = time()
        # candidate_local_embeddings = database_local_embeddings[m][nn_indices]
        # candidate_keypoints = local_map_embeddings_keypoints[m][nn_indices]
        # fitness_list = sgv_fn(query_local_embeddings[n][query_idx], candidate_local_embeddings, candidate_keypoints, d_thresh=0.4)
        # topk_rerank = np.flip(np.asarray(fitness_list).argsort())
        # topk_rerank_indices = copy.deepcopy(nn_indices)
        # topk_rerank_indices[:topk] = nn_indices[topk_rerank]
        # t_rerank = time() - tick
        # intermediate_metrics['t_rr'].append(t_rerank)

        # delta_rerank = query_position - database_positions[m][topk_rerank_indices]
        # euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)
        ########################################################################

        # Count true positives and 1% retrieved for different radius and NN number
        intermediate_metrics['tp'] = {r: [intermediate_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in radius}
        # intermediate_metrics['tp_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(num_neighbors)] for r in self.radius}
        intermediate_metrics['opr'] = {r: intermediate_metrics['opr'][r] + (1 if (euclid_dist[:opr_threshold] <= r).any() else 0) for r in radius}
        # intermediate_metrics['opr_rr'] = {r: intermediate_metrics['opr_rr'][r] + (1 if (euclid_dist_rr[:threshold] <= r).any() else 0) for r in self.radius}
        intermediate_metrics['RR'] = {r: intermediate_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in radius}
        # intermediate_metrics['RR_rr'] = {r: intermediate_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}

        # LOCAL DESCRIPTOR EVALUATION
        # Do the evaluation only if the nn pose is within distance threshold
        # Otherwise the overlap is too small to get reasonable results
        # (evaluation continues if standard OR re-ranked nn pose is within thresh meters)
        if euclid_dist[0] > params.local.max_eval_threshold:  # and euclid_dist_rr[0] > local_max_eval_threshold:
            continue

        # Cache query and nn idx for metric loc eval
        metric_loc_pairs_list['Initial'].append((query_idx, nn_indices[0]))
        # pairs_dict['Re-Ranked'].append(query_idx, topk_rerank_indices[0]))  # TODO: Enable when re-ranking implemented

    # Calculate mean global metrics
    global_metrics['recall'] = {r: [intermediate_metrics['tp'][r][nn] / num_evaluated for nn in range(num_neighbors)] for r in radius}
    # global_metrics['recall_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] / num_evaluated for nn in range(num_neighbors)] for r in radius}
    global_metrics['recall@1'] = {r: global_metrics['recall'][r][0] for r in radius}
    # global_metrics['recall@1_rr'] = {r: global_metrics['recall_rr'][r][0] for r in radius}
    global_metrics['recall@1%'] = {r: intermediate_metrics['opr'][r] / num_evaluated for r in radius}
    # global_metrics['recall@1%_rr'] = {r: intermediate_metrics['opr_rr'][r] / num_evaluated for r in radius}
    global_metrics['MRR'] = {r: np.mean(np.asarray(intermediate_metrics['RR'][r])) for r in radius}
    # global_metrics['MRR_rr'] = {r: np.mean(np.asarray(intermediate_metrics['RR_rr'][r])) for r in radius}

    msg = 'Place Recognition Results:'
    recall = global_metrics['recall']
    for radius in recall.keys():
        msg += f"\n  Radius: {radius} [m]:\n    Recall@N: "
        for ii, x in enumerate(recall[radius]):
                msg += f"{x:0.3f}, "
                if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                    msg += "\n              "
        msg += "\n    Recall@1%: {:0.3f}".format(global_metrics['recall@1%'][radius])
        msg += '\n    MRR: {:0.3f}'.format(global_metrics['MRR'][radius])
    print(msg)

    eval_dataloader = make_eval_dataloader_6DOF(
        params, query_set, database_set, metric_loc_pairs_list[EVAL_MODE]
    )
    return eval_dataloader, query_local_embeddings, database_local_embeddings

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
    if os.path.splitext(args.weights)[1] == '.ckpt':
        state = torch.load(args.weights)
        model.load_state_dict(state['model_state_dict'])
    else:  # .pt or .pth
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)
    model.eval()

    logging_level = 'DEBUG' if params.debug else 'INFO'
    create_logger(log_file=None, logging_level=logging_level)

    if params.local.eval_num_workers > 0:
        params.local.eval_num_workers = 2
    (
        dataloader,
        query_local_embeddings,
        database_local_embeddings,
    ) = get_eval_pairs_dataloader(params, model, model_name, device)
    # dataloader.shuffle = True  # NOTE: setting to true will invalidate metric_loc_pairs_list order
    
    return params, device, model, dataloader, query_local_embeddings, database_local_embeddings,

def main():
    (
        params,
        device,
        model,
        dataloader,
        query_local_embeddings,
        database_local_embeddings,
    ) = setup_model()

    # Get evaluator
    metric_loc_evaluator = Evaluator(params)
    metric_loc_pairs_list = dataloader.dataset.pairs_list

    for ii, local_batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        query_idx, nn_idx = metric_loc_pairs_list[ii]
        if len(args.idx_list) > 0 and query_idx not in args.idx_list:
            continue
        local_batch['anc_local_feats'] = {
            'coarse': query_local_embeddings['coarse'][query_idx],
            'fine': query_local_embeddings['fine'][query_idx],
        }
        local_batch['pos_local_feats'] = {
            'coarse': database_local_embeddings['coarse'][nn_idx],
            'fine': database_local_embeddings['fine'][nn_idx],
        }
        local_batch = to_device(local_batch, device, non_blocking=True, construct_octree_neigh=True)
        with torch.inference_mode():
            output_dict = model(local_batch)[0]

        batch_temp = {'transform': local_batch['transform'][0]}  # temp fix since loss func expects a single batch item
        eval_metrics = release_cuda(metric_loc_evaluator(output_dict, batch_temp))

        # Skip successes
        if args.failures and eval_metrics['RR'] > 0:
            continue

        anc_points_coarse = release_cuda(output_dict['anc_points_coarse'])
        anc_points_fine = release_cuda(output_dict['anc_points_fine'])
        anc_point_to_node = release_cuda(output_dict['anc_point_to_node'])
        anc_feats_coarse = release_cuda(output_dict['anc_feats_coarse'], to_numpy=True)
        anc_feats_coarse_pre_refinement = release_cuda(output_dict['anc_feats_coarse_pre_refinement'], to_numpy=True)
        anc_feats_fine = release_cuda(output_dict['anc_feats_fine'], to_numpy=True)
        pos_points_coarse = release_cuda(output_dict['pos_points_coarse'], to_numpy=True)
        pos_points_fine = release_cuda(output_dict['pos_points_fine'])
        pos_point_to_node = release_cuda(output_dict['pos_point_to_node'])
        pos_feats_coarse = release_cuda(output_dict['pos_feats_coarse'])
        pos_feats_coarse_pre_refinement = release_cuda(output_dict['pos_feats_coarse_pre_refinement'], to_numpy=True)
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
        print(log_str, flush=True)

        save_dir_ii = None
        if args.save_dir is not None:
            save_dir_ii = os.path.join(args.save_dir, f'{query_idx}')
            os.makedirs(save_dir_ii, exist_ok=True)
            # Save stats to dir for future reference:
            with open(os.path.join(save_dir_ii, 'stats.txt'), 'w') as f:
                f.write(log_str)

        if not args.registration_only:
            visualise_coarse_correspondences(
                anc_points_coarse=anc_points_coarse,
                pos_points_coarse=pos_points_coarse,
                anc_points_fine=anc_points_fine,
                pos_points_fine=pos_points_fine,
                node_corr_indices=node_corr_indices.numpy(),
                gt_node_corr_indices=gt_node_corr_indices.numpy(),
                transform=T_gt.numpy(),
                anc_point_to_node=anc_point_to_node,
                pos_point_to_node=pos_point_to_node,
                anc_feats_coarse=anc_feats_coarse,
                pos_feats_coarse=pos_feats_coarse,
                translate=[0,0,50],
                zoom=args.zoom,
                plot_coarse=True,  # Plot keypoints as spheres
                # coarse_colourmode='patch',
                # coarse_colourmode='tsne',
                coarse_colourmode='umap',
                save_dir=save_dir_ii,
                disable_animation=args.disable_animation,
                non_interactive=args.non_interactive,
            )

            visualise_similarity(
                anc_points_fine=anc_points_fine,
                pos_points_fine=pos_points_fine,
                transform=T_gt.numpy(),
                anc_point_to_node=anc_point_to_node,
                pos_point_to_node=pos_point_to_node,
                anc_feats_coarse=anc_feats_coarse_pre_refinement,
                pos_feats_coarse=pos_feats_coarse_pre_refinement,
                translate=[0,0,50],
                zoom=args.zoom,
                # coarse_colourmode='tsne',
                coarse_colourmode='umap',
                save_dir=save_dir_ii,
                disable_animation=args.disable_animation,
                non_interactive=args.non_interactive,
            )

            visualise_fine_correspondences(
                anc_points_fine=anc_points_fine,
                pos_points_fine=pos_points_fine,
                anc_corr_points=anc_corr_points,
                pos_corr_points=pos_corr_points,
                corr_scores=corr_scores,
                transform=T_gt.numpy(),
                score_threshold=0.0,
                anc_feats_fine=anc_feats_fine,
                pos_feats_fine=pos_feats_fine,
                translate=[0,0,50],
                zoom=args.zoom,
                colourmode='umap',
                save_dir=save_dir_ii,
                disable_animation=args.disable_animation,
                non_interactive=args.non_interactive,
            )

        # Best (initial) correspondence from LGw
        if T_best_corr is not None:
            visualise_LGR_initial_registration(
                anc_corr_points=best_anc_corr_points,
                pos_corr_points=best_pos_corr_points,
                corr_scores=best_corr_scores,
                transform=T_best_corr.numpy(),
                # transform=T_gt.numpy(),
                # translate=[0,0,3],
                zoom=args.zoom,
                angle=-380,  # ~deg*10
                save_dir=save_dir_ii,
                non_interactive=args.non_interactive,
            )

        # Estimated TF
        visualise_registration(
            anc_points_fine=anc_points_fine,
            pos_points_fine=pos_points_fine,
            transform=T_estimated.numpy(),
            zoom=args.zoom,
            save_dir=save_dir_ii,
            filename='registration',
            non_interactive=args.non_interactive,
        )

        # Ground truth TF (with ICP, if enabled in params)
        visualise_registration(
            anc_points_fine=anc_points_fine,
            pos_points_fine=pos_points_fine,
            transform=T_gt.numpy(),
            zoom=args.zoom,
            save_dir=save_dir_ii,
            filename='registration_GT',
            non_interactive=args.non_interactive,
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
    parser.add_argument('--registration_only', action='store_true', help='Only visualise registration')
    parser.add_argument('--idx_list', type=int, nargs='+', default=[], help='Only visualise given list of indices (ordered by val dataloader)')
    parser.add_argument('--zoom', type=float, default=0.55, help='Zoom level for open3d viewer')
    parser.add_argument('--disable_animation', action='store_true', help='Disables animations')
    parser.add_argument('--non_interactive', action='store_true', help='Saves visualisations instantly, then closes viewer.')
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings to disk')
    parser.add_argument('--load_embeddings', action='store_true',
                        help=('Load embeddings from disk. Note this script will only check if '
                              'weights paths match, not if the configs used match.'))
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print(f'Dataset split idx: {args.dataset_idx}')
    print(f'Database split idx: {args.database_split_idx}')
    print(f'Query split idx: {args.query_split_idx}')
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
    print('Registration only: {}'.format(args.registration_only))
    print('Idx list: {}'.format(args.idx_list))
    if args.failures and len(args.idx_list) > 0:
        print('WARNING: Ignoring `--failures` as `--idx_list` was specified')
        args.failures = False
    print('Zoom {}'.format(args.zoom))
    print('Disable animation: {}'.format(args.disable_animation))
    print('Non interactive: {}'.format(args.non_interactive))
    if args.save_embeddings and args.load_embeddings:
        print('[WARNING] Both save_embeddings AND load_embeddings specified, which is redundant. '
              'Will proceed without saving embeddings.')
        args.save_embeddings = False
    print(f'Save embeddings: {args.save_embeddings}')
    print(f'Load embeddings: {args.load_embeddings}')
    print('Debug mode: {}'.format(args.debug))
    print('')

    set_seed()  # Seed RNG
    main()