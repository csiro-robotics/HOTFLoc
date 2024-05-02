# Warsaw University of Technology

import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree
import ocnn

from dataset.base_datasets import EvaluationTuple, TrainingDataset
from dataset.augmentation import TrainSetTransform
from dataset.pointnetvlad.pnv_train import PNVTrainingDataset
from dataset.pointnetvlad.pnv_train import TrainTransform as PNVTrainTransform
from dataset.AboveUnder.AboveUnder_train import AboveUnderTrainingDataset
from dataset.AboveUnder.AboveUnder_train import TrainTransform as AboveUnderTrainTransform
from dataset.AboveUnder.AboveUnder_train import ValTransform as AboveUnderValTransform
from dataset.samplers import BatchSampler
from misc.utils import TrainingParams
from dataset.base_datasets import PointCloudLoader
from dataset.pointnetvlad.pnv_raw import PNVPointCloudLoader
from dataset.AboveUnder.AboveUnder_raw import AboveUnderPointCloudLoader


def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    if 'AboveUnder' in dataset_type or 'WildPlaces' in dataset_type:
        return AboveUnderPointCloudLoader
    else:
        return PNVPointCloudLoader()


def make_datasets(params: TrainingParams, validation: bool = True):
    # Create training and validation datasets
    datasets = {}
    train_set_transform = TrainSetTransform(params.set_aug_mode)

    # PoinNetVLAD datasets (RobotCar and Inhouse)
    # PNV datasets have their own transform
    if 'AboveUnder' in params.dataset_name or 'WildPlaces' in params.dataset_name:
        train_transform = AboveUnderTrainTransform(params.aug_mode, normalize_points=params.normalize_points)
        datasets['train'] = AboveUnderTrainingDataset(params.dataset_folder, params.train_file,
                                                      transform=train_transform, set_transform=train_set_transform,
                                                      load_octree=params.load_octree, octree_depth=params.octree_depth)
        if validation:
            val_transform = AboveUnderValTransform(normalize_points=params.normalize_points)
            datasets['val'] = AboveUnderTrainingDataset(params.dataset_folder, params.val_file,
                                                        transform=val_transform,
                                                        load_octree=params.load_octree, octree_depth=params.octree_depth)
    else:
        train_transform = PNVTrainTransform(params.aug_mode)
        datasets['train'] = PNVTrainingDataset(params.dataset_folder, params.train_file,
                                               transform=train_transform, set_transform=train_set_transform,
                                               load_octree=params.load_octree, octree_depth=params.octree_depth)
        if validation:
            datasets['val'] = PNVTrainingDataset(params.dataset_folder, params.val_file)

    return datasets


def make_collate_fn(dataset: TrainingDataset, quantizer, batch_split_size=None,
                    load_octree=False):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    # batch_split_size: if not None, splits the batch into a list of multiple mini-batches with batch_split_size elems
    # octree: if True, loads octree in batch instead of sparse tensor
    def collate_fn(data_list):
        # Constructs a batch object
        data = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        if not load_octree:
            clouds = data
            if dataset.set_transform is not None:
                # Apply the same transformation on all dataset elements
                lens = [len(cloud) for cloud in clouds]
                clouds = torch.cat(clouds, dim=0)
                clouds = dataset.set_transform(clouds)
                clouds = clouds.split(lens)

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        if load_octree:
            if batch_split_size is None or batch_split_size == 0:
                octrees = ocnn.octree.merge_octrees(data)
                # NOTE: remember to construct the neighbor indices
                octrees.construct_all_neigh()
                batch = {'octree': octrees}

            else:
                # Split the batch into chunks
                batch = []
                for i in range(0, len(data), batch_split_size):
                    temp = data[i:i + batch_split_size]
                    octrees_temp = ocnn.octree.merge_octrees(temp)
                    # NOTE: remember to construct the neighbor indices
                    octrees_temp.construct_all_neigh()
                    minibatch = {'octree': octrees_temp}
                    batch.append(minibatch)
        else:
            # Convert to polar (when polar coords are used) and quantize
            # Use the first value returned by quantizer
            coords = [quantizer(e)[0] for e in clouds]

            if batch_split_size is None or batch_split_size == 0:
                coords = ME.utils.batched_coordinates(coords)
                # Assign a dummy feature equal to 1 to each point
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                batch = {'coords': coords, 'features': feats}

            else:
                # Split the batch into chunks
                batch = []
                for i in range(0, len(coords), batch_split_size):
                    temp = coords[i:i + batch_split_size]
                    c = ME.utils.batched_coordinates(temp)
                    f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                    minibatch = {'coords': c, 'features': f}
                    batch.append(minibatch)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and negatives_mask which are
        # batch_size x batch_size boolean tensors
        #return batch, positives_mask, negatives_mask, torch.tensor(sampled_positive_ndx), torch.tensor(relative_poses)
        return batch, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders(params: TrainingParams, validation=True):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, validation=validation)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    quantizer = params.model_params.quantizer
    train_collate_fn = make_collate_fn(datasets['train'],  quantizer, params.batch_split_size,
                                       load_octree=params.load_octree)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=params.num_workers,
                                     pin_memory=True)
    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], quantizer, params.batch_split_size,
                                         load_octree=params.load_octree)
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders


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
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e

