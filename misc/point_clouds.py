import copy
import os
from typing import Optional

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def draw_registration_result(source, target, transformation):
    if not isinstance(source, o3d.geometry.PointCloud):
        source = make_open3d_point_cloud(source)
    if not isinstance(target, o3d.geometry.PointCloud):
        target = make_open3d_point_cloud(target)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def draw_pc(pc):
    if not isinstance(pc, o3d.geometry.PointCloud):
        pc = make_open3d_point_cloud(pc)
    pc = copy.deepcopy(pc)
    pc.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([pc],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def plot_points(points: np.ndarray, show=True):
    """
    Plots a point cloud using matplotlib. Colormap is based on z height.

    Args:
        points (ndarray): Point cloud of shape (N, 3), with (x,y,z) coords.
    """    
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(*points.T, c=points.T[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal', adjustable='box')
    if show:
        plt.show()


def icp(
    anchor_pc,
    positive_pc,
    transform: Optional[np.ndarray] = None,
    point2plane=False,
    gicp=False,
    inlier_dist_threshold: float = 1.2,
    max_iteration: int = 200,
    voxel_size: Optional[float] = 0.1,
):
    assert not(point2plane and gicp), "Choose either point2plane or gicp method, not both"
    # transform: initial alignment transform
    if transform is not None:
        transform = transform.astype(float)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(anchor_pc)
    if voxel_size is not None:
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(positive_pc)
    if voxel_size is not None:
        pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    if gicp:
        transform_estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        registration_fn = o3d.pipelines.registration.registration_generalized_icp
    else:
        if point2plane:
            pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
            pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
            transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        registration_fn = o3d.pipelines.registration.registration_icp

    if transform is not None:
        reg_p2p = registration_fn(pcd1, pcd2, inlier_dist_threshold, transform,
                                  estimation_method=transform_estimation,
                                  criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    else:
        reg_p2p = registration_fn(pcd1, pcd2, inlier_dist_threshold,
                                  estimation_method=transform_estimation,
                                  criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse


def fast_global_registration(
    anchor_pc,
    positive_pc,
    inlier_dist_threshold: float = 0.2,
    max_iteration: int = 64,
    voxel_size: Optional[float] = 0.4,
):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(anchor_pc)
    if voxel_size is not None:
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(positive_pc)
    if voxel_size is not None:
        pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    pcd1_fpfh = compute_fpfh(pcd1, voxel_size=voxel_size)
    pcd2_fpfh = compute_fpfh(pcd2, voxel_size=voxel_size)
    
    # Recommended threshold is 0.5x voxel size
    fgr_option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=inlier_dist_threshold,
            iteration_number=max_iteration,
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pcd1, pcd2, pcd1_fpfh, pcd2_fpfh, option=fgr_option,
    )

    return result.transformation, result.fitness, result.inlier_rmse


def compute_fpfh(pcd, voxel_size: Optional[float] = 0.4):
    # Radius normal and feature size based on Open3D recommendations:
    # https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Extract-geometric-feature 
    if voxel_size is None:
        voxel_size = 0.4
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_fpfh


def make_open3d_feature(data, dim, npts):
    feature = o3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.cpu().numpy().astype('d').transpose()
    return feature


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def preprocess_pointcloud(
    pc,
    remove_zero_points: bool = False,
    min_x: float = None,
    max_x: float = None,
    min_y: float = None,
    max_y: float = None,
    min_z: float = None,
    max_z: float = None,
):
    if remove_zero_points:
        mask = np.all(np.isclose(pc, 0.), axis=1)
        pc = pc[~mask]
    if min_x is not None:
        mask = pc[:, 0] > min_x
        pc = pc[mask]
    if max_x is not None:
        mask = pc[:, 0] <= max_x
        pc = pc[mask]
    if min_y is not None:
        mask = pc[:, 1] > min_y
        pc = pc[mask]
    if max_y is not None:
        mask = pc[:, 1] <= max_y
        pc = pc[mask]
    if min_z is not None:
        mask = pc[:, 2] > min_z
        pc = pc[mask]
    if max_z is not None:
        mask = pc[:, 2] <= max_z
        pc = pc[mask]
    return pc


class PointCloudLoader:
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
        pc = self.read_pc(file_pathname)
        assert pc.shape[1] == 3

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]

        return pc

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")