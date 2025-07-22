# Warsaw University of Technology

import numpy as np
from typing import List, Sequence, Dict
import time
from itertools import repeat
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree
import ocnn
from ocnn.octree import Octree, Points

from dataset.base_datasets import EvaluationTuple, TrainingDataset, Training6DOFDataset, EvalDataset
from dataset.augmentation import TrainTransform, TrainSetTransform, Train6DOFTransform, ValTransform, Val6DOFTransform
from dataset.samplers import BatchSampler, BatchSampler6DOF
from misc.utils import TrainingParams


def make_datasets(params: TrainingParams, local=False, validation=True):
    # Create training and validation datasets
    datasets = {}
    train_set_transform = TrainSetTransform(params.set_aug_mode, random_rot_theta=params.random_rot_theta)

    train_transform = TrainTransform(
        params.aug_mode, normalize_points=params.normalize_points,
        scale_factor=params.scale_factor, unit_sphere_norm=params.unit_sphere_norm,
        zero_mean=params.zero_mean, random_rot_theta=params.random_rot_theta
    )
    datasets['global_train'] = TrainingDataset(
        params.dataset_folder, params.dataset_name, params.train_file, transform=train_transform,
        set_transform=train_set_transform, load_octree=params.load_octree,
        octree_depth=params.octree_depth, full_depth=params.full_depth,
        coordinates=params.model_params.coordinates
    )
    if validation:
        val_transform = ValTransform(
            normalize_points=params.normalize_points, scale_factor=params.scale_factor,
            unit_sphere_norm=params.unit_sphere_norm, zero_mean=params.zero_mean
        )
        datasets['global_val'] = TrainingDataset(
            params.dataset_folder, params.dataset_name, params.val_file, transform=val_transform,
            load_octree=params.load_octree, octree_depth=params.octree_depth,
            full_depth=params.full_depth, coordinates=params.model_params.coordinates
        )
    if local:
        local_train_transform = Train6DOFTransform(
            params.local_aug_mode, normalize_points=params.normalize_points,
            scale_factor=params.scale_factor, unit_sphere_norm=params.unit_sphere_norm,
            zero_mean=params.zero_mean, random_rot_theta=params.random_rot_theta
        )
        datasets['local_train'] = Training6DOFDataset(
            params.dataset_folder, params.dataset_name, params.train_file, local_transform=local_train_transform,
            load_octree=params.load_octree, octree_depth=params.octree_depth,
            full_depth=params.full_depth, coordinates=params.model_params.coordinates
        )
        if validation:
            local_val_transform = Val6DOFTransform(
                normalize_points=params.normalize_points, scale_factor=params.scale_factor,
                unit_sphere_norm=params.unit_sphere_norm, zero_mean=params.zero_mean
            )
            datasets['local_val'] = Training6DOFDataset(
                params.dataset_folder, params.dataset_name, params.val_file, local_transform=local_val_transform,
                load_octree=params.load_octree, octree_depth=params.octree_depth,
                full_depth=params.full_depth, coordinates=params.model_params.coordinates
            )

    return datasets


def create_batch(clouds: Sequence[torch.Tensor], quantizer, params: TrainingParams):
    """
    Util function to create batches in correct format from an input list of 
    point clouds.

    Args:
        clouds (Sequence[Tensor]): Sequence of point clouds of shape (N, 3).
        quantizer (Optional): If using MinkLoc, quantizer for sparse quantization.
            If using OctFormer, coordinate system converter.
        params (TrainingParams): Training parameters for the model.
    """
    if params.load_octree:
        if quantizer is not None:            
            clouds = [quantizer(e) for e in clouds]
        octrees = []
        # Convert to ocnn Points object, then create Octree
        for cloud in clouds:
            cloud_points_obj = Points(cloud)
            octree = Octree(params.octree_depth, params.full_depth)
            octree.build_octree(cloud_points_obj)
            octrees.append(octree)                        
        octrees_merged = ocnn.octree.merge_octrees(octrees)
        # NOTE: remember to construct the neighbor indices
        octrees_merged.construct_all_neigh()
        batch = {'octree': octrees_merged}
    else:
        coords = [quantizer(e)[0] for e in clouds]
        coords = ME.utils.batched_coordinates(coords)
        # Assign a dummy feature equal to 1 to each point
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': coords, 'features': feats}
    return batch


def make_collate_fn(dataset: TrainingDataset, quantizer, params: TrainingParams):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    # batch_split_size: if not None, splits the batch into a list of multiple mini-batches with batch_split_size elems
    # octree: if True, loads octree in batch instead of sparse tensor
    def collate_fn(data_list):
        if params.verbose:
            tic = time.time()

        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        
        # clouds = data
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            lens = [len(cloud) for cloud in clouds]
            clouds_merged = torch.cat(clouds, dim=0)
            clouds_merged = dataset.set_transform(clouds_merged)
            clouds = clouds_merged.split(lens)

        # Compute positives and negatives mask
        positives_mask = np.asarray([in_sorted_array_vectorised(labels, dataset.get_positives(label)) for label in labels])
        negatives_mask = np.asarray([in_sorted_array_vectorised(labels, dataset.get_non_negatives(label)) for label in labels])
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask).logical_not()

        # Generate batches in correct format for MinkLoc/OctFormer
        if params.batch_split_size is None or params.batch_split_size == 0:
            batch = create_batch(clouds, quantizer, params)
        else:
            # Split the batch into chunks
            batch = []
            for i in range(0, len(clouds), params.batch_split_size):
                temp = clouds[i:i + params.batch_split_size]
                minibatch = create_batch(temp, quantizer, params)
                batch.append(minibatch)

        if params.verbose:
            toc = time.time()
            print(f"[INFO] Collating batch done in {toc-tic:.2f}s for {dataset.query_filepath}", flush=True)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and negatives_mask which are
        # batch_size x batch_size boolean tensors
        #return batch, positives_mask, negatives_mask, torch.tensor(sampled_positive_ndx), torch.tensor(relative_poses)
        return batch, positives_mask, negatives_mask

    return collate_fn


def make_collate_fn_6DOF(dataset: Training6DOFDataset, quantizer, params: TrainingParams):
    """
    Collation function for local batches. Returns batch + 6DOF relative transform + 
    normalization shift and scale parameters.
    """
    def collate_fn(data_list):
        # Constructs a batch object
        anchor_clouds = [e[0] for e in data_list]
        positive_clouds = [e[1] for e in data_list]
        rel_transforms = [e[2] for e in data_list]
        anchor_shift_and_scale = [e[3] for e in data_list]
        positive_shift_and_scale = [e[4] for e in data_list]
        len_batch = []
        for batch_id, _ in enumerate(anchor_clouds):
            N1 = anchor_clouds[batch_id].shape[0]
            N2 = positive_clouds[batch_id].shape[0]
            len_batch.append([N1, N2])

        # Generate anchor and positive batches
        anchor_batch = create_batch(anchor_clouds, quantizer, params)
        positive_batch = create_batch(positive_clouds, quantizer, params)
        # Concatenate point coordinates
        anchor_xyz = torch.cat(anchor_clouds, dim=0)
        positive_xyz = torch.cat(positive_clouds, dim=0)
        # Stack transforms
        trans_batch = torch.stack(rel_transforms, dim=0)

        # Returns:
        # Anc and pos point clouds, anc and pos batch, relative transformations, length per batch, normalization shift and scale params
        return {
            "anc_pcd": anchor_xyz, "pos_pcd": positive_xyz, "anc_batch": anchor_batch,
            "pos_batch": positive_batch, "T_gt": trans_batch, "len_batch": len_batch,
            "anc_shift_and_scale": anchor_shift_and_scale, "pos_shift_and_scale": positive_shift_and_scale
        }

    return collate_fn


def make_dataloaders(params: TrainingParams, local=False, validation=True):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements

    :param train_params:
    :param model_params:
    :return:
    :rtype: dict
    """
    datasets = make_datasets(params, local=local, validation=validation)

    dataloaders = {}
    train_sampler = BatchSampler(datasets['global_train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    quantizer = params.model_params.quantizer
    train_collate_fn = make_collate_fn(datasets['global_train'], quantizer, params)
    dataloaders['global_train'] = DataLoader(datasets['global_train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=params.num_workers,
                                     pin_memory=True)
    if validation and 'global_val' in datasets:
        val_collate_fn = make_collate_fn(datasets['global_val'], quantizer, params)
        val_sampler = BatchSampler(datasets['global_val'], batch_size=params.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloaders['global_val'] = DataLoader(datasets['global_val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    if local:
        train_collate_fn_loc = make_collate_fn_6DOF(datasets['local_train'], quantizer, params)
        train_sampler_loc = BatchSampler6DOF(datasets['local_train'], batch_size=params.local_batch_size)
        dataloaders['local_train'] = DataLoader(datasets['local_train'], batch_sampler=train_sampler_loc, 
                                                collate_fn=train_collate_fn_loc,
                                                num_workers=params.num_workers, pin_memory=True)
        if validation and 'local_val' in datasets:
            val_collate_fn_loc = make_collate_fn_6DOF(datasets['local_val'], quantizer, params)
            val_sampler_loc = BatchSampler6DOF(datasets['local_val'], batch_size=params.local_batch_size)
            dataloaders['local_val'] = DataLoader(datasets['local_val'], batch_sampler=val_sampler_loc,
                                                  collate_fn=val_collate_fn_loc,
                                                  num_workers=params.num_workers, pin_memory=True)
    else:  # Endlessly return None
        dataloaders['local_train'] = repeat(None)
        if validation:
            dataloaders['local_val'] = repeat(None)

    return dataloaders


def make_eval_dataset(params: TrainingParams, data_set: Dict, local=False):
    """
    Create dataset class for a single sequence.
    """
    if local:
        raise NotImplementedError()
    else:
        val_transform = ValTransform(
            normalize_points=params.normalize_points, scale_factor=params.scale_factor,
            unit_sphere_norm=params.unit_sphere_norm, zero_mean=params.zero_mean
        )
        dataset = EvalDataset(
            params.dataset_folder, params.dataset_name, data_set,
            transform=val_transform, load_octree=params.load_octree,
            coordinates=params.model_params.coordinates
        )

    return dataset


def make_eval_collate_fn(quantizer, params: TrainingParams):
    """
    Custom collate function for evaluation dataloader. Only returns batches.
    """
    def collate_fn(data_list):
        if params.verbose:
            tic = time.time()

        # Generate batches in correct format for MinkLoc/OctFormer
        batch = create_batch(data_list, quantizer, params)

        if params.verbose:
            toc = time.time()
            print(f"[INFO] Collating batch done in {toc-tic:.2f}s", flush=True)

        return batch

    return collate_fn


def make_eval_dataloader(params: TrainingParams, data_set: Dict, local=False):
    """
    Creates dataloader suitable for evaluate.py script.
    """
    quantizer = params.model_params.quantizer
    eval_dataset = make_eval_dataset(params, data_set, local=local)
    eval_collate_fn = make_eval_collate_fn(quantizer, params)
    eval_dataloader = DataLoader(eval_dataset, batch_size=params.val_batch_size,
                                 shuffle=False, pin_memory=True, num_workers=params.num_workers,
                                 collate_fn=eval_collate_fn)
    return eval_dataloader


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] "
          f"radius")
    return filtered_query_set


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    """DEPRECATED
    ---
    Much slower than vectorised version
    """
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e


def in_sorted_array_vectorised(query_array : np.ndarray,
                               sorted_array: np.ndarray) -> List[bool]:
    """Check if each element of `query_array` is in `sorted_array`.

    Args:
        query_array (np.ndarray): Array of integers to check.
        sorted_array (np.ndarray): Sorted array of integers.

    Returns:
        np.ndarray: Boolean array where each element indicates if the corresponding
            element in `query_array` is in `sorted_array`.
    """
    query_array = np.asarray(query_array)
    sorted_array = np.asarray(sorted_array)
    
    # Return immediately if no elems in either array
    if len(sorted_array) == 0 or len(query_array) == 0:
        return np.full(query_array.shape, False, bool)
    
    # Find indices where elements of query_array would fit in sorted_array
    indices = np.searchsorted(sorted_array, query_array)
    
    # Clip indices to stay within bounds of sorted_array
    indices = np.clip(indices, 0, len(sorted_array) - 1)
    
    # Check if the elements of query_array match the elements in sorted_array
    # at the computed indices
    return sorted_array[indices] == query_array


############### CODE FOR TESTING in_sorted_array ###############
# def run_in_sorted_array(labels, array):
#     # temp for debugging
#     l = []
#     for e in labels:
#         l.append(in_sorted_array(e, array))

#     return l

# if __name__ == "__main__":
#     from timeit import timeit
#     sorted_array = np.arange(0,2048,16)
#     values = np.random.randint(0, 2048, size=2048)

#     out1 = run_in_sorted_array(values, sorted_array)
#     print(torch.tensor(out1))
#     out2 = in_sorted_array_vectorised(values, sorted_array)
#     print(torch.tensor(out2))
#     print(all(out1 == out2))

#     assert(all(out1 == out2)), "Vectorised func output != standard func output!"
    
#     print("Timing...")
    
#     time_searchsorted = timeit(
#         stmt="run_in_sorted_array(values, sorted_array)",
#         setup="from __main__ import run_in_sorted_array, sorted_array, values",
#         number=1000,  # Number of repetitions
#     )
#     print(f"Standard searchsorted: {time_searchsorted:.6f} seconds")
    
#     time_custom_searchsorted = timeit(
#         stmt="in_sorted_array_vectorised(values, sorted_array)",
#         setup="from __main__ import in_sorted_array_vectorised, sorted_array, values",
#         number=1000,  # Number of repetitions
#     )
#     print(f"Vectorised searchsorted: {time_custom_searchsorted:.6f} seconds")
