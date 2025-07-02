# Base dataset classes, inherited by dataset-specific classes
import os
import pickle
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.pointnetvlad.pnv_raw import PNVPointCloudLoader
from dataset.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader
from dataset.point_clouds import PointCloudLoader


class TrainingTuple:
    # Tuple describing an element for training/validation, with optional pose for metric localisation
    def __init__(self, id: int, timestamp: float, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray, pose: np.ndarray = None,
                 positive_poses: Dict[int, np.ndarray] = None):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # non_negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        # pose: optional pose as SE(3) matrix
        # positives_poses: optional relative poses of positive examples (ideally refined using ICP)
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position
        self.pose = pose
        self.positive_poses = positive_poses


class EvaluationTuple:
    # Tuple describing an evaluation set element, with optional pose for metric localisation
    def __init__(self, timestamp: float, rel_scan_filepath: str, position: np.ndarray, pose: np.ndarray = None):
        # position: x, y position in meters
        # pose: optional pose as SE(3) matrix
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)

        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position
        self.pose = pose

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position, self.pose


class TrainingDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_type: str, query_filename: str,
                 transform=None, set_transform=None, load_octree=False,
                 octree_depth=11, full_depth=2, coordinates='cartesian'):
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.coordinates = coordinates
        self.load_octree = load_octree
        self.octree_depth = octree_depth
        self.full_depth = full_depth

        # NOTE: Virga env requires 'datasets' module stored as 'dataset' to avoid
        # conflict. Below will check if a pickle was saved with the wrong module,
        # and load TrainingTuple from the correct path for this environment.
        self.queries: Dict[int, TrainingTuple] = CustomUnpickler(open(self.query_filepath, 'rb')).load()
        print('{} queries in the dataset'.format(len(self)))

        self.pc_loader: PointCloudLoader = get_pointcloud_loader(self.dataset_type)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        query_pc = self.pc_loader(file_pathname)
        data = torch.tensor(query_pc, dtype=torch.float)
        if self.transform is not None:
            data = self.transform(data)
        if self.load_octree:
            # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
            mask = torch.all(abs(data) <= 1.0, dim=1)
            data = data[mask]
            # Also ensure this will hold if converting coordinate systems
            if self.coordinates == 'cylindrical':
                data_norm = torch.linalg.norm(data[:, :2], dim=1)[:, None]
                mask = torch.all(data_norm <= 1.0, dim=1)
                data = data[mask]
            # elif self.coordinates == 'spherical':  # TODO: if spherical is used
            #     data_norm = torch.linalg.norm(data, dim=1)[:, None]
            #     mask = torch.all(data_norm <= 1.0, dim=1)
        return data, ndx

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class Training6DOFDataset(TrainingDataset):
    """
    Dataset wrapper for for 6dof estimation.
    """
    def __init__(self, dataset_path: str, dataset_type: str, query_filename: str,
                 local_transform=None, **kwargs):
        super().__init__(dataset_path, dataset_type, query_filename, **kwargs)
        self.local_transform = local_transform

    def __getitem__(self, ndx):
        # TODO: Consider adding gravity alignment as a step here
        query_shift_and_scale = None
        positive_shift_and_scale = None
        # pose is a global coordinate system pose 3x4 R|T matrix
        query_pc, _ = super().__getitem__(ndx)

        # get random positive
        positives = self.get_positives(ndx)
        positive_idx = np.random.choice(positives, 1)[0]
        positive_pc, _ = super().__getitem__(positive_idx)

        # get relative pose from global poses
        if self.queries[ndx].positive_poses is None:
            raise ValueError("Invalid training tuple, ensure 6DOF poses are present")
        transform = torch.tensor(self.queries[ndx].positives_poses[positive_idx].copy())
        
        if self.local_transform is not None:
            # Apply only normalization and jitter to the query point cloud
            query_pc, query_shift_and_scale, _ = self.local_transform(query_pc, ignore_rot_and_trans=True)
            # Apply random transform to the positive point cloud
            # TODO: Verify that augmented transform correctly registers to query
            positive_pc, positive_shift_and_scale, aug_tf = self.local_transform(positive_pc)
            transform = aug_tf @ transform

        return query_pc, positive_pc, transform, query_shift_and_scale, positive_shift_and_scale


class EvalDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_type: str, data_set_dict: Dict,
                 transform=None, load_octree=False, coordinates='cartesian'):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        assert len(data_set_dict) > 0, "Empty dict"
        self.tuple_dict = data_set_dict
        self.transform = transform
        self.coordinates = coordinates
        self.load_octree = load_octree

        self.pc_loader: PointCloudLoader = get_pointcloud_loader(self.dataset_type)

    def __len__(self):
        return len(self.tuple_dict)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.dataset_path, self.tuple_dict[ndx]['query'])
        query_pc = self.pc_loader(file_pathname)
        data = torch.tensor(query_pc, dtype=torch.float)
        if self.transform is not None:
            data = self.transform(data)
        if self.load_octree:
            # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
            mask = torch.all(abs(data) <= 1.0, dim=1)
            data = data[mask]
            # Also ensure this will hold if converting coordinate systems
            if self.coordinates == 'cylindrical':
                data_norm = torch.linalg.norm(data[:, :2], dim=1)[:, None]
                mask = torch.all(data_norm <= 1.0, dim=1)
                data = data[mask]
            # elif self.coordinates == 'spherical':  # TODO: if spherical is used
            #     data_norm = torch.linalg.norm(data, dim=1)[:, None]
            #     mask = torch.all(data_norm <= 1.0, dim=1)
        return data


class EvaluationSet:
    """NOT IN USE CURRENTLY  
    Evaluation set consisting of map and query elements
    """
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions


def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    if 'WildPlaces' in dataset_type:
        return CSWildPlacesPointCloudLoader()
    elif dataset_type in ['Oxford', 'CSCampus3D']:
        return PNVPointCloudLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")


class CustomUnpickler(pickle.Unpickler):
    """
    Custom unpickler class to handle training pickles that were created in the
    live repo of HOTFormerLoc. This repo is the dev repo, and due to the Virga
    environment setup to get MinkowskiEngine working alongside HOTFormerLoc, the
    'datasets' module had to be renamed to 'dataset' to avoid conflict with a
    pip module called 'datasets' in the system python package. Yes there is
    probably a more elegant solution to prevent this, but this class offers a
    quick and dirty solution for my dev branch.
    """
    def find_class(self, module, name):
        old_module = "datasets.base_datasets"
        class_name = "TrainingTuple"
        # In the below line, you can ignore the module and only check the name
        # if you're sure there is only one implementation of the class you're
        # replacing.
        if module == old_module and name == class_name:
            # Redirect to the correct TrainingTuple class
            return TrainingTuple
        return super().find_class(module, name)