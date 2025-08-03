# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
# Adapted to process samples in batches by Ethan Griffiths (QUT & CSIRO Data61).

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import logging
import torch
import tqdm

from models.model_factory import model_factory
from misc.logger import create_logger
from misc.utils import TrainingParams, load_pickle, save_pickle
from misc.torch_utils import set_seed, to_device, release_cuda
from dataset.dataset_utils import make_eval_dataloader
from eval.utils import get_query_database_splits

def evaluate(model, device, params: TrainingParams, log: bool = False,
             model_name: str = 'model', show_progress: bool = False, 
             save_embeddings: bool = False, load_embeddings: bool = False):
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
            if "5m-pickles/" in database_file:  # wild-places
                location_name = location_name.split('/')[-1]
                temp = temp.split('/')[-1]
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
        temp = evaluate_dataset(model, device, params, database_sets, query_sets, location_name,
                                log=log, model_name=model_name, show_progress=show_progress,
                                save_embeddings=save_embeddings, load_embeddings=load_embeddings)
        stats[location_name] = temp
        ave_one_percent_recall.append(temp['ave_one_percent_recall'])
        ave_recall.append(temp['ave_recall'])
        ave_mrr.append(temp['ave_mrr'])
        
    # Compute average stats
    stats['average'] = {'ave_one_percent_recall': np.mean(ave_one_percent_recall),
                        'ave_recall': np.mean(ave_recall, axis=0),
                        'ave_mrr': np.mean(ave_mrr)}
    return stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets, query_sets,
                     location_name: str, log: bool = False, model_name: str = 'model',
                     show_progress: bool = False, save_embeddings: bool = False,
                     load_embeddings: bool = False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []
    mrr = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    if show_progress:
        logging.info(f'{"Loading" if load_embeddings else "Computing"} database embeddings')
    for ii, data_set in enumerate(database_sets):
        temp_embeddings = None
        if len(data_set) > 0:
            if load_embeddings:
                temp_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'database_{ii}')
            else:
                temp_embeddings = get_latent_vectors(model, data_set, device, params, show_progress)
            if save_embeddings:
                save_embeddings_to_file(temp_embeddings, model_name, params.dataset_name, location_name, f'database_{ii}')
        database_embeddings.append(temp_embeddings)

    if show_progress:
        logging.info(f'{"Loading" if load_embeddings else "Computing"} query embeddings')
    for jj, data_set in enumerate(query_sets):
        temp_embeddings = None
        if len(data_set) > 0:
            if load_embeddings:
                temp_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'query_{jj}')
            else:
                temp_embeddings = get_latent_vectors(model, data_set, device, params, show_progress)
            if save_embeddings:
                save_embeddings_to_file(temp_embeddings, model_name, params.dataset_name, location_name, f'query_{jj}')
        query_embeddings.append(temp_embeddings)

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
            pair_recall, pair_opr, pair_mrr = get_recall(i, j, database_embeddings,
                                                         query_embeddings, query_sets,
                                                         database_sets, log=log,
                                                         model_name=model_name)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            mrr.append(pair_mrr)

    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    ave_mrr = np.mean(mrr)
    stats = {'ave_one_percent_recall': ave_one_percent_recall,
             'ave_recall': ave_recall,
             'ave_mrr': ave_mrr}
    return stats


def get_latent_vectors(model, data_set, device, params: TrainingParams,
                       show_progress: bool = False):
    # Adapted from original PointNetVLAD code
    if len(data_set) == 0:
        return None

    ### NOTE: Disabled so that eval can be tested during training debug mode
    # if params.debug:
    #     embeddings = np.random.rand(len(data_set), 256)
    #     return embeddings

    # Create dataloader for data_set
    dataloader = make_eval_dataloader(params, data_set)
    embeddings = None
    model.eval()
    with tqdm.tqdm(total=len(dataloader.dataset), disable=(not show_progress)) as pbar:
        for ii, batch in enumerate(dataloader):
            batch = to_device(batch, device, non_blocking=True, construct_octree_neigh=True)
            temp_embedding = compute_embedding(model, batch)
            if embeddings is None:
                embeddings = np.zeros((len(data_set), temp_embedding.shape[1]), dtype=temp_embedding.dtype)
            embeddings[ii*params.val_batch_size:(ii*params.val_batch_size + len(temp_embedding))] = temp_embedding
            pbar.update(len(temp_embedding))
    
    return embeddings


def compute_embedding(model, batch):
    with torch.inference_mode():
        # Compute global descriptor
        y = model(batch, global_only=True)
        embedding = release_cuda(y['global'], to_numpy=True)
        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return embedding


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets,
               log=False, model_name: str = 'model'):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors
    recall_idx = []

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        if log:
            # Log false positives (returned as the first element)
            # Check if there's a false positive returned as the first element
            if indices[0][0] not in true_neighbors:
                fp_ndx = indices[0][0]
                fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0, 0]  # Distance in embedding space
                fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
                                        (query_details['easting'] - fp['easting']) ** 2)
                # Find the first true positive
                tp = None
                for k in range(len(indices[0])):
                    if indices[0][k] in true_neighbors:
                        closest_pos_ndx = indices[0][k]
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
            for k in range(min(len(indices[0]), 5)):
                is_match = indices[0][k] in true_neighbors
                e_ndx = indices[0][k]
                e = database_sets[m][e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
                e_emb_dist = distances[0][k]
                world_dist = np.sqrt((query_details['northing'] - e['northing']) ** 2 +
                                        (query_details['easting'] - e['easting']) ** 2)
                s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
            s += '\n'
            out_top5_file_name = f"{model_name}_log_search_results.txt"
            with open(out_top5_file_name, "a") as f:
                f.write(s)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                recall_idx.append(j+1)
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    mrr = np.mean(1/np.array(recall_idx))*100
    return recall, one_percent_recall, mrr


def print_eval_stats(stats):
    msg = 'Eval Results'
    for database_name in stats:
        msg += '\nDataset: {}\n'.format(database_name)
        t = 'Avg. top 1% recall: {:.2f}   Avg. MRR: {:.2f}   Avg. recall @N:\n'
        msg += t.format(
            stats[database_name]["ave_one_percent_recall"],
            stats[database_name]["ave_mrr"],
        )
        msg += (stats[database_name]['ave_recall'])
    logging.info(msg)


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            s += f"\n[{ds}]\n"
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            ave_mrr = stats[ds]['ave_mrr']
            s += "AR@1%: {:0.2f}, AR@1: {:0.2f}, MRR: {:0.2f}, AR@N:\n".format(ave_1p_recall, ave_recall, ave_mrr)
            s += str(stats[ds]['ave_recall'])

        # NOTE: below is redundant as average is now stored in stats dict
        # mean_1p_recall = np.mean(ave_1p_recall_l)
        # mean_recall = np.mean(ave_recall_l)
        # s += "\n\nMean AR@1%: {:0.2f}, Mean AR@1: {:0.2f}\n\n".format(mean_1p_recall, mean_recall)
        s += "\n------------------------------------------------------------------------\n\n"
        f.write(s)


def save_embeddings_to_file(global_embeddings, model_name: str, dataset_name: str,
                            location_name: str, set_name: str = "database"):
    save_dir = os.path.join(os.path.dirname(__file__), "embeddings", dataset_name, location_name, f"model_{model_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    global_embeddings_file = os.path.join(save_dir, f"{set_name}_global_embeddings.pickle")
    save_pickle(global_embeddings, global_embeddings_file)


def load_embeddings_from_file(model_name: str, dataset_name: str,
                              location_name: str, set_name: str = "database"):
    load_dir = os.path.join(os.path.dirname(__file__), "embeddings", dataset_name, location_name, f"model_{model_name}")
    if not os.path.exists(load_dir):
        raise FileNotFoundError("No saved embeddings found for model. Run with --save_embeddings first.")
    else:
        global_embeddings_file = os.path.join(load_dir, f"{set_name}_global_embeddings.pickle")
        global_embeddings = load_pickle(global_embeddings_file)
        assert (len(global_embeddings) > 0), (
            "Saved descriptors are corrupted, rerun with `save_descriptors` enabled"
        )
    return global_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true', help="Log false positives and top-5 retrievals")
    parser.set_defaults(log=False)
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings to disk')
    parser.add_argument('--load_embeddings', action='store_true',
                        help=('Load embeddings from disk. Note this script will only check if '
                              'weights paths match, not if the configs used match.'))

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
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
    print('Determinism: Enabled')

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

    stats = evaluate(model, device, params, args.log, model_name, show_progress=True,
                     save_embeddings=args.save_embeddings,
                     load_embeddings=args.load_embeddings)
    print_eval_stats(stats)

    pnv_write_eval_stats(f"pnv_{params.dataset_name}_results.txt", prefix, stats)

