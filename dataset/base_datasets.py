# Base dataset classes, inherited by dataset-specific classes
import os
import pickle
import time
import logging
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.pointnetvlad.pnv_raw import PNVPointCloudLoader
from dataset.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader
from dataset.mulran.mulran_raw import MulranPointCloudLoader
from dataset.mulran.utils import relative_pose as mulran_relative_pose
from dataset.southbay.southbay_raw import SouthbayPointCloudLoader
from misc.point_clouds import PointCloudLoader, icp, two_stage_icp
from misc.poses import relative_pose as base_relative_pose
from misc.poses import height_offset_removal, gravity_align_pc_with_pose, invert_pose


class TrainingTuple:
    # Tuple describing an element for training/validation, with optional pose for metric localisation
    def __init__(
        self,
        id: int,
        timestamp: float,
        rel_scan_filepath: str,
        positives: np.ndarray,
        non_negatives: np.ndarray,
        position: np.ndarray,
        pose: np.ndarray = None,
        positives_poses: Dict[int, np.ndarray] = None,
    ):
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
        self.positives_poses = positives_poses


class EvaluationTuple:
    # Tuple describing an evaluation set element, with optional pose for metric localisation
    def __init__(
        self,
        timestamp: float,
        rel_scan_filepath: str,
        position: np.ndarray,
        pose: np.ndarray = None,
    ):
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


def clip_points(data: torch.Tensor, coordinates: str):
    """
    Clip point coordinates to [-1, 1] range. Ensures this operation holds
    for cylindrical coordinates too.
    """
    assert coordinates in ['cartesian', 'cylindrical']
    # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
    mask = torch.all(abs(data) <= 1.0, dim=1)
    data = data[mask]
    # Also ensure this will hold if converting coordinate systems
    if coordinates == 'cylindrical':
        data_norm = torch.linalg.norm(data[:, :2], dim=1)[:, None]
        mask = torch.all(data_norm <= 1.0, dim=1)
        data = data[mask]
    # elif self.coordinates == 'spherical':  # TODO: if spherical is used
    #     data_norm = torch.linalg.norm(data, dim=1)[:, None]
    #     mask = torch.all(data_norm <= 1.0, dim=1)
    return data

class TrainingDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_type: str,
        query_filename: str,
        transform=None,
        set_transform=None,
        load_octree=False,
        octree_depth=11,
        full_depth=2,
        coordinates="cartesian",
        is_cross_source_dataset=False,
        prioritise_cross_source=False,
        remove_height_offset=False,
        gravity_align=False,
    ):
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), f'Cannot access dataset path: {dataset_path}'
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), f'Cannot access query file: {self.query_filepath}'
        self.transform = transform
        self.set_transform = set_transform
        self.coordinates = coordinates
        self.load_octree = load_octree
        self.octree_depth = octree_depth
        self.full_depth = full_depth
        if prioritise_cross_source:
            assert is_cross_source_dataset
        self.is_cross_source_dataset = is_cross_source_dataset
        self.prioritise_cross_source = prioritise_cross_source
        self.remove_height_offset = remove_height_offset
        self.gravity_align = gravity_align
        self.clip_octree_points = True  # clip point coordinates to [-1, 1] range (if load_octree is True)

        # NOTE: Virga env requires 'datasets' module stored as 'dataset' to avoid
        # conflict. Below will check if a pickle was saved with the wrong module,
        # and load TrainingTuple from the correct path for this environment.
        self.queries: Dict[int, TrainingTuple] = CustomUnpickler(open(self.query_filepath, "rb")).load()
        message = f'{len(self)} queries in the dataset'
        # Pre-compute ground and aerial indices
        if self.is_cross_source_dataset:
            self.ground_ndx = {idx for idx, elem in self.queries.items() if 'ground' in elem.rel_scan_filepath}
            self.aerial_ndx = {idx for idx, elem in self.queries.items() if 'aerial' in elem.rel_scan_filepath}
            message += f' ({len(self.ground_ndx)} ground, {len(self.aerial_ndx)} aerial)'
        logging.info(message)

        self.pc_loader = get_pointcloud_loader(self.dataset_type)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        query_pc = self.pc_loader(file_pathname)
        data = torch.tensor(query_pc, dtype=torch.float)
        shift_and_scale = None
        tf_gravity_align = torch.eye(4)
        tf_height_offset_removal = torch.eye(4)
        if self.gravity_align:
            data, tf_gravity_align = gravity_align_pc_with_pose(data, self.queries[ndx].pose)
        if self.remove_height_offset:
            data, tf_height_offset_removal = height_offset_removal(data)
        if self.transform is not None:
            data, shift_and_scale = self.transform(data)
        tf_composed = tf_height_offset_removal @ tf_gravity_align
        if self.load_octree and self.clip_octree_points:
            data = clip_points(data, self.coordinates)
        assert data.size(0) > 0
        return data, ndx, shift_and_scale, tf_composed

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

    def get_cross_source_positives(
        self,
        ndx: int,
        positives: Optional[List[int]],
    ) -> List[int]:
        """Returns list of positives captured from a different source to the query."""
        if positives is None:
            positives = self.get_positives(ndx)
        if 'ground' in self.queries[ndx].rel_scan_filepath:
            cross_source_positives = self.aerial_ndx.intersection(positives)
        elif 'aerial' in self.queries[ndx].rel_scan_filepath:
            cross_source_positives = self.ground_ndx.intersection(positives)
        else:
            raise ValueError('Dataset does not support ground/aerial cross-source')
        return list(cross_source_positives)


class Training6DOFDataset(TrainingDataset):
    """
    Dataset wrapper for 6DOF estimation. Loads pairs of positive point clouds
    with relative transform and normalization parameters.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_type: str,
        query_filename: str,
        local_transform=None,
        icp=False,
        icp_use_gicp=True,
        icp_inlier_dist_threshold: float = 0.2,
        icp_max_iteration: int = 100,
        icp_voxel_size: Optional[float] = None,
        icp_two_stage: bool = False,
        icp_two_stage_inlier_dist_threshold: float = 5.5,
        icp_two_stage_max_iteration: int = 50,
        icp_two_stage_voxel_size: Optional[float] = 0.5,
        **kwargs,
    ):
        super().__init__(dataset_path, dataset_type, query_filename, **kwargs)
        self.clip_octree_points = False  # We do this step after local transforms
        self.local_transform = local_transform
        self.icp = icp
        self.icp_use_gicp = icp_use_gicp
        self.icp_inlier_dist_threshold = icp_inlier_dist_threshold
        self.icp_max_iteration = icp_max_iteration
        self.icp_voxel_size = icp_voxel_size
        self.icp_two_stage = icp_two_stage
        self.icp_two_stage_inlier_dist_threshold = icp_two_stage_inlier_dist_threshold
        self.icp_two_stage_max_iteration = icp_two_stage_max_iteration
        self.icp_two_stage_voxel_size = icp_two_stage_voxel_size
        if self.dataset_type.lower() == 'mulran':
            self.relative_pose = mulran_relative_pose
        else:
            self.relative_pose = base_relative_pose

    def __getitem__(self, ndx):
        # TODO: Consider adding gravity alignment as a step here
        query_shift_and_scale = None
        positive_shift_and_scale = None
        # pose is a global coordinate system pose 3x4 R|T matrix
        query_pc, _, _, query_applied_tf = super().__getitem__(ndx)

        # get random positive
        positives = self.get_positives(ndx)
        if self.prioritise_cross_source:
            cross_source_positives = self.get_cross_source_positives(ndx, positives)
            if len(cross_source_positives) > 0:
                positives = cross_source_positives
        positive_ndx = np.random.choice(positives, 1)[0]
        positive_pc, _, _, positive_applied_tf = super().__getitem__(positive_ndx)

        # get relative pose from global poses
        if self.queries[ndx].positives_poses is not None:
            transform = torch.tensor(
                self.queries[ndx].positives_poses[positive_ndx],
                dtype=query_pc.dtype,
            )
        else:
            transform = torch.tensor(
                self.relative_pose(self.queries[ndx].pose, self.queries[positive_ndx].pose),
                dtype=query_pc.dtype,
            )

        # Correct height offsets and/or gravity alignment in gt relative transform
        transform = positive_applied_tf @ transform @ invert_pose(query_applied_tf)

        # ######################## TEMP FOR DEBUGGING ########################
        # query_pc_orig = query_pc.clone()
        # positive_pc_orig = positive_pc.clone()
        # transform_orig = transform.clone()
        # ####################################################################

        # Ensure alignment with icp
        if self.icp:
            # tic = time.perf_counter()
            if self.icp_two_stage:
                transform_icp, fitness_icp, inlier_rmse_icp = two_stage_icp(
                    query_pc.numpy().astype(float),
                    positive_pc.numpy().astype(float),
                    transform.numpy(),
                    gicp=self.icp_use_gicp,
                    inlier_dist_threshold=self.icp_inlier_dist_threshold,
                    max_iteration=self.icp_max_iteration,
                    voxel_size=self.icp_voxel_size,
                    two_stage_inlier_dist_threshold=self.icp_two_stage_inlier_dist_threshold,
                    two_stage_max_iteration=self.icp_two_stage_max_iteration,
                    two_stage_voxel_size=self.icp_two_stage_voxel_size,
                )
            else:
                transform_icp, fitness_icp, inlier_rmse_icp = icp(
                    query_pc.numpy().astype(float),
                    positive_pc.numpy().astype(float),
                    transform.numpy(),
                    gicp=self.icp_use_gicp,
                    inlier_dist_threshold=self.icp_inlier_dist_threshold,
                    max_iteration=self.icp_max_iteration,
                    voxel_size=self.icp_voxel_size,
                )
            # msg = f"[ICP] Fitness: {fitness_icp:.4f} -- Inlier RMSE: {inlier_rmse_icp:.4f} -- {time.perf_counter() - tic:.4f}s"
            # logging.debug(msg)
            transform = torch.tensor(transform_icp, dtype=transform.dtype)

        if self.local_transform is not None:
            # Apply only normalization and jitter to the query point cloud
            query_pc, query_shift_and_scale, _ = self.local_transform(query_pc, ignore_rot_and_trans=True)
            # Apply random transform to the positive point cloud
            positive_pc, positive_shift_and_scale, aug_tf = self.local_transform(positive_pc)
            transform = aug_tf @ transform

        ########################################################################
        # VISUALISATIONS FOR DEBUGGING
        ########################################################################
        # from misc.point_clouds import draw_registration_result
        # query_pc_denorm = self.local_transform.normalization_transform.unnormalize(query_pc, query_shift_and_scale)
        # positive_pc_denorm = self.local_transform.normalization_transform.unnormalize(positive_pc, positive_shift_and_scale)
        # # draw_registration_result(query_pc_denorm, query_pc_orig, np.eye(4))
        # # draw_registration_result(positive_pc_denorm, positive_pc_orig, np.eye(4))
        # # draw_registration_result(positive_pc_denorm, positive_pc_orig, np.linalg.inv(aug_tf))
        # # draw_registration_result(query_pc_orig, positive_pc_orig, np.eye(4))
        # # draw_registration_result(query_pc_orig, positive_pc_orig, transform_orig)
        # # # draw_registration_result(query_pc_orig, positive_pc_orig, transform_fgr)
        # # draw_registration_result(query_pc_orig, positive_pc_orig, transform_icp)
        # # draw_registration_result(query_pc_denorm, positive_pc_denorm, np.eye(4))
        # draw_registration_result(query_pc_denorm, positive_pc_denorm, transform)  # this should be correct, but currently isnt (for K-01 1624327938.8356152.pcd and K-02 1624318871.8437989.pcd, actual pose files seem incorrect)
        # # draw_registration_result(query_pc_denorm, positive_pc_denorm, aug_tf @ transform_icp)
        ########################################################################

        # Now clip point coordinates after normalization is done
        if self.load_octree:
            query_pc = clip_points(query_pc, self.coordinates)
            positive_pc = clip_points(positive_pc, self.coordinates)

        return query_pc, query_shift_and_scale, positive_pc, positive_shift_and_scale, transform


class EvalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_type: str,
        data_set_dict: Dict,
        transform=None,
        load_octree=False,
        coordinates="cartesian",
        remove_height_offset=False,
        gravity_align=False,
    ):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        assert len(data_set_dict) > 0, "Empty dict"
        self.data_set_dict = data_set_dict
        self.transform = transform
        self.load_octree = load_octree
        self.coordinates = coordinates
        self.remove_height_offset = remove_height_offset
        self.gravity_align = gravity_align
        self.clip_octree_points = True  # clip point coordinates to [-1, 1] range (if load_octree is True)
        self.max_points_egonn = 120000

        self.pc_loader: PointCloudLoader = get_pointcloud_loader(self.dataset_type)

    def __len__(self):
        return len(self.data_set_dict)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.dataset_path, self.data_set_dict[ndx]['query'])
        query_pc = self.pc_loader(file_pathname)
        data = torch.tensor(query_pc, dtype=torch.float)
        # NOTE: EgoNN seems to raise CUDA errors for batches with very large point clouds
        #       (even without raising OOM), so limiting them here as a temp workaround
        if not self.load_octree:
            indices = torch.randperm(len(data))[:self.max_points_egonn]
            data = data[indices]
        shift_and_scale = None
        tf_gravity_align = torch.eye(4)
        tf_height_offset_removal = torch.eye(4)
        if self.gravity_align:
            data, tf_gravity_align = gravity_align_pc_with_pose(data, self.data_set_dict[ndx]['pose'])
        if self.remove_height_offset:
            data, tf_height_offset_removal = height_offset_removal(data)
        if self.transform is not None:
            data, shift_and_scale = self.transform(data, random_seed=ndx)
        tf_composed = tf_height_offset_removal @ tf_gravity_align
        assert data.size(0) > 0
        return data, shift_and_scale, tf_composed


class EvalRerankingDataset(EvalDataset):
    """
    Dataset wrapper for reranking evaluation. Loads query and top-N candidates
    as a single batch from a pre-computed list of indices.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_type: str,
        query_set: Dict,
        database_set: Dict,
        query_nn_list: List,
        **kwargs,
    ):
        super().__init__(dataset_path, dataset_type, query_set, **kwargs)
        self.database_dataset = EvalDataset(dataset_path, dataset_type, database_set, **kwargs)
        self.query_nn_list = query_nn_list

    def __len__(self):
        return len(self.query_nn_list)

    def __getitem__(self, ndx):
        query_ndx, nn_indices = self.query_nn_list[ndx]
        pc_list = []
        shift_and_scale_list = []
        query_pc, query_shift_and_scale, _ = super().__getitem__(query_ndx)
        pc_list.append(query_pc)
        shift_and_scale_list.append(query_shift_and_scale)
        for nn_idx in nn_indices:
            nn_pc, nn_shift_and_scale, _ = self.database_dataset[nn_idx]
            pc_list.append(nn_pc)
            shift_and_scale_list.append(nn_shift_and_scale)

        return pc_list, shift_and_scale_list


class Eval6DOFDataset(EvalDataset):
    """
    Dataset wrapper for 6DOF estimation. Loads pairs of positive point clouds
    with relative transform and normalization parameters.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_type: str,
        query_set: Dict,
        database_set: Dict,
        pairs_list: List,
        local_transform=None,
        icp=False,
        icp_use_gicp=True,
        icp_inlier_dist_threshold: float = 0.2,
        icp_max_iteration: int = 100,
        icp_voxel_size: Optional[float] = None,
        icp_two_stage: bool = False,
        icp_two_stage_inlier_dist_threshold: float = 8.0,
        icp_two_stage_max_iteration: int = 50,
        icp_two_stage_voxel_size: Optional[float] = 0.8,
        **kwargs,
    ):
        super().__init__(dataset_path, dataset_type, query_set, **kwargs)
        self.pos_dataset = EvalDataset(dataset_path, dataset_type, database_set, **kwargs)
        self.clip_octree_points = False  # We do this step after local transforms
        self.pos_dataset.clip_octree_points = False
        self.pairs_list = pairs_list
        self.local_transform = local_transform
        self.icp = icp
        self.icp_use_gicp = icp_use_gicp
        self.icp_inlier_dist_threshold = icp_inlier_dist_threshold
        self.icp_max_iteration = icp_max_iteration
        self.icp_voxel_size = icp_voxel_size
        self.icp_two_stage = icp_two_stage
        self.icp_two_stage_inlier_dist_threshold = icp_two_stage_inlier_dist_threshold
        self.icp_two_stage_max_iteration = icp_two_stage_max_iteration
        self.icp_two_stage_voxel_size = icp_two_stage_voxel_size
        if self.dataset_type.lower() == 'mulran':
            self.relative_pose = mulran_relative_pose
        else:
            self.relative_pose = base_relative_pose

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, ndx):
        query_shift_and_scale = None
        positive_shift_and_scale = None

        query_ndx, positive_ndx = self.pairs_list[ndx]
        query_pc, _, query_applied_tf = super().__getitem__(query_ndx)
        positive_pc, _, positive_applied_tf = self.pos_dataset[positive_ndx]

        # get relative pose from global poses
        transform = torch.tensor(
            self.relative_pose(
                self.data_set_dict[query_ndx]['pose'],
                self.pos_dataset.data_set_dict[positive_ndx]['pose'],
            ),
            dtype=query_pc.dtype,
        )

        # Correct height offsets and/or gravity alignment in gt relative transform
        transform = positive_applied_tf @ transform @ invert_pose(query_applied_tf)

        # ######################## TEMP FOR DEBUGGING ########################
        # query_pc_orig = query_pc.clone()
        # positive_pc_orig = positive_pc.clone()
        # transform_orig = transform.clone()
        # ####################################################################

        # Ensure alignment with icp
        if self.icp:
            # tic = time.perf_counter()
            if self.icp_two_stage:
                transform_icp, fitness_icp, inlier_rmse_icp = two_stage_icp(
                    query_pc.numpy().astype(float),
                    positive_pc.numpy().astype(float),
                    transform.numpy(),
                    gicp=self.icp_use_gicp,
                    inlier_dist_threshold=self.icp_inlier_dist_threshold,
                    max_iteration=self.icp_max_iteration,
                    voxel_size=self.icp_voxel_size,
                    two_stage_inlier_dist_threshold=self.icp_two_stage_inlier_dist_threshold,
                    two_stage_max_iteration=self.icp_two_stage_max_iteration,
                    two_stage_voxel_size=self.icp_two_stage_voxel_size,
                )
            else:
                transform_icp, fitness_icp, inlier_rmse_icp = icp(
                    query_pc.numpy().astype(float),
                    positive_pc.numpy().astype(float),
                    transform.numpy(),
                    gicp=self.icp_use_gicp,
                    inlier_dist_threshold=self.icp_inlier_dist_threshold,
                    max_iteration=self.icp_max_iteration,
                    voxel_size=self.icp_voxel_size,
                )
            # msg = f"[ICP] Fitness: {fitness_icp:.4f} -- Inlier RMSE: {inlier_rmse_icp:.4f} -- {time.perf_counter() - tic:.4f}s"
            # logging.debug(msg)
            transform = torch.tensor(transform_icp, dtype=transform.dtype)
        
        if self.local_transform is not None:
            # Apply normalization 
            query_pc, query_shift_and_scale, _ = self.local_transform(query_pc, ignore_rot_and_trans=True, random_seed=ndx)
            positive_pc, positive_shift_and_scale, _ = self.local_transform(positive_pc, ignore_rot_and_trans=True, random_seed=positive_ndx)

        ########################################################################
        # VISUALISATIONS FOR DEBUGGING
        ########################################################################
        # from misc.point_clouds import draw_registration_result
        # query_pc_denorm = self.local_transform.normalization_transform.unnormalize(query_pc, query_shift_and_scale)
        # positive_pc_denorm = self.local_transform.normalization_transform.unnormalize(positive_pc, positive_shift_and_scale)
        # # draw_registration_result(query_pc_denorm, query_pc_orig, np.eye(4))
        # # draw_registration_result(positive_pc_denorm, positive_pc_orig, np.eye(4))
        # # draw_registration_result(query_pc_orig, positive_pc_orig, np.eye(4))
        # # draw_registration_result(query_pc_orig, positive_pc_orig, transform_orig)
        # # draw_registration_result(query_pc_orig, positive_pc_orig, transform_icp)
        # # draw_registration_result(query_pc_denorm, positive_pc_denorm, np.eye(4))
        # draw_registration_result(query_pc_denorm, positive_pc_denorm, transform)  # this should be correct, but currently isnt (for K-01 1624327938.8356152.pcd and K-02 1624318871.8437989.pcd, actual pose files seem incorrect)
        ########################################################################

        # Now clip point coordinates after normalization is done
        if self.load_octree:
            query_pc = clip_points(query_pc, self.coordinates)
            positive_pc = clip_points(positive_pc, self.coordinates)

        return query_pc, query_shift_and_scale, positive_pc, positive_shift_and_scale, transform


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
    elif dataset_type.lower() == 'mulran':
        return MulranPointCloudLoader()
    elif dataset_type.lower() == 'southbay':
        return SouthbayPointCloudLoader()
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