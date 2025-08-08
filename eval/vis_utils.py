"""
Utility functions for visualising HOTFormerLoc and octrees.

Ethan Griffiths (Data61, Pullenvale)
"""
from typing import List, Optional, Union
import logging

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d

from models.octree import OctreeT
from misc.point_clouds import make_open3d_point_cloud

def submap_distance(q1, q2) -> float:
    """
    Returns the distance between two submaps based on easting and northing
    """
    q1_pos = np.array([q1['easting'], q1['northing']])
    q2_pos = np.array([q2['easting'], q2['northing']])
    return np.linalg.norm(q2_pos - q1_pos)

def rowwise_cosine_sim(a: Tensor, b: Tensor, eps=1e-8):
    """
    Computes row-wise cosine similarity between two tensors, with added eps for
    numerical stability.
    """
    assert a.ndim == 2 and b.ndim == 2
    assert a.size(1) == b.size(1)
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def off_diagonal(x: Tensor):
    """
    Get off diagonal elements of square matrix x.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def print_token_similarity(token_dict, token_type: str = 'Local'):
    """
    Takes a dict of tokens (N, C) with keys representing octree depth, and for
    each set of tokens at each depth, computes and prints the average similarity
    between tokens.
    """
    pyramid_depths = list(token_dict.keys())
    fig = plt.figure(figsize=(8,7))
    # for j, depth_j in enumerate(pyramid_depths):
    #     sim = cosine_sim_matrix(token_dict[depth_j], token_dict[depth_j])
    #     print(f"{token_type} token average similarity - depth {depth_j} ({len(sim)} tokens total):")
    #     print(f"Overall similarity between tokens: {sim.mean():.3f} (min similarity {sim.min():.3f})")
    #     # print(f"Average per row:\n{sim.mean(0)}")
    #     ax = fig.add_subplot(1,3,j+1)
    #     ax.boxplot(sim.mean(0))
    #     ax.set_title(f"{token_type} token similarities - depth {depth_j}")
    # ax.xticks(range(len(pyramid_depths)), pyramid_depths)

    sims_rowwise = []
    for j, depth_j in enumerate(pyramid_depths):
        sim = rowwise_cosine_sim(token_dict[depth_j], token_dict[depth_j])
        print(f"{token_type} token average similarity - depth {depth_j} ({len(sim)} tokens total):")
        print(f"Overall similarity between tokens: {sim.mean():.3f} (min similarity {sim.min():.3f})")
        # print(f"Average per row:\n{sim.mean(0)}")
        sims_rowwise.append(sim.mean(0))  # avg per row
        # sims_rowwise.append(sim.flatten())  # avg overall (incl. diagonal)
        
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(sims_rowwise)
    ax.set_title(f"{token_type} token similarities")
    ax.set_xticks(np.arange(len(pyramid_depths)) + 1, pyramid_depths)
    ax.set_xlabel("Octree Depth")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim([-0.4, 1])

def remove_rt_attn_padding(
    relay_token_attn_map: Tensor,
    octree: OctreeT,
    batch_idx: int,
) -> Tensor:
    """
    Removes padding tokens from relay token attention map. Expects a single 
    attention head as input.

    Args:
        relay_token_attn_map (Tensor): Tensor of shape (N, N) containing attn map.
        octree (OctreeT): Octree corresponding to attn map.
        batch_idx (int): Batch index of attn map.
    """
    rt_row_mask_idx = octree.rt_attn_mask[batch_idx,0] != octree.invalid_mask_value
    relay_token_attn_map = relay_token_attn_map[rt_row_mask_idx][:,rt_row_mask_idx]
    return relay_token_attn_map

def get_rt_boundary_idx(
    octree: OctreeT,
    pyramid_depths: List[int],
    return_cumsum: bool = True
):
    """
    Returns the idx of the final relay token for each pyramid level, minus the
    padding elements. Also optionally returns the cumsum of these indices. 
    """
    rt_boundary_idx = [
        (octree.batch_num_windows[depth_j].item()
         - torch.count_nonzero(octree.ct_batch_idx[depth_j]).item())
        for depth_j in pyramid_depths
    ]
    if not return_cumsum:
        return rt_boundary_idx
    else: 
        rt_boundary_idx_cumsum = np.cumsum(rt_boundary_idx).tolist()
        return rt_boundary_idx, rt_boundary_idx_cumsum

def colourise_points(
    values: ndarray, colourmap_name='viridis',
) -> ndarray:
    """
    Colourise a point cloud based on scalar values using a matplotlib colourmap.
    
    Args:
        values: numpy array of shape (N, 1) containing scalar values to colourise each point by
        colourmap_name: name of the matplotlib colourmap to use (default: 'viridis')
    
    Returns:
        colours: numpy array of shape (N, 3) containing RGB values in range [0, 1]
    """
    # Normalize values to range [0, 1]
    if np.max(values) > np.min(values):
        normalised = (values - np.min(values)) / (np.max(values) - np.min(values))
    else:
        # Handle the case where all points have the same value
        normalised = np.zeros_like(values)
    
    # Get the colormap from matplotlib
    cmap = plt.get_cmap(colourmap_name)
    
    # Convert normalized values to colors
    colours = cmap(normalised)[:, :3]  # Only take RGB, discard alpha
    return colours

def colourise_points_by_height(
    points: ndarray, colourmap_name='viridis',
) -> ndarray:
    """
    Colourise a point cloud based on z values using a matplotlib colourmap.
    
    Args:
        points: numpy array of shape (N, 3) containing 3D points (last dim MUST be height)
        colourmap_name: name of the matplotlib colourmap to use (default: 'viridis')
    
    Returns:
        colours: numpy array of shape (N, 3) containing RGB values in range [0, 1]
    """
    assert points.ndim == 2 and points.shape[1] == 3
    # Extract z-values (height)
    z_values = points[:, 2]
    
    # Convert normalized values to colors
    colours = colourise_points(z_values, colourmap_name=colourmap_name)
    return colours

def get_colours_with_tsne(data: ndarray) -> ndarray:
    r"""
    TODO: Use this func for colourisation  
    Use t-SNE to project high-dimension feats to rgb

    Args:
        data (ndarray): (N, C)

    Returns:
        colors (ndarray): (N, 3)
    """
    tsne = TSNE(n_components=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data).reshape(-1)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    normalized_tsne_results = (tsne_results - tsne_min) / (tsne_max - tsne_min)
    colors = plt.cm.Spectral(normalized_tsne_results)[:, :3]
    return colors

def colourise_points_by_similarity(
    embeddings: ndarray, mode: str = 'tsne', return_explained_variance=False,
) -> Union[ndarray, List[ndarray]]:
    """
    Colourise a point cloud based on similarity of local features. Uses t-SNE or
    PCA to compute the colourisation.
    
    Args:
        embeddings: numpy array of shape (N, D) containing embeddings for each point
        mode: tSNE or PCA projection
        return_explained_variance: if using PCA, returns explained variance of each component
    
    Returns:
        colours: numpy array of shape (N, 3) containing RGB values in range [0, 1]
    """
    assert mode.lower() in ('tsne', 'pca'), "mode must be 'tsne' or 'pca'"
    assert embeddings.ndim == 2
    eps = 1e-8
    if mode.lower() == 'tsne':
        # NOTE: perplexity is not meant to be lower than n_samples, so may need to adjust this
        #       for shallow octree levels with only a handful of octants
        colours = get_colours_with_tsne(embeddings)
    elif mode.lower() == 'pca':
        pca = PCA(n_components=3)
        colours = pca.fit_transform(embeddings)
        pca_explained_variance = pca.explained_variance_ratio_

    # Normalize to [0, 1]
    colours = (colours - colours.min(0)) / (colours.max(0) - colours.min(0) + eps)

    if mode.lower() == 'pca' and return_explained_variance:
        return colours, pca_explained_variance
    return colours

def create_heatmap(values: Tensor, ticklabels: Optional[List] = None,
                   min_value: Optional[float] = None, title: Optional[str] = None) -> plt.Figure:
    CMAP = 'viridis'
    vmin = None
    if ticklabels is None:
        ticklabels = 'auto'
    fig = plt.figure(figsize=(6,5))
    # Clip masked values to prevent them overpowering the attn map
    if min_value is not None: 
        vmin = values[values > min_value].min().item()
    ax = sns.heatmap(values, cmap=CMAP, vmin=vmin, xticklabels=ticklabels, yticklabels=ticklabels)
    if title is not None:
        ax.set_title(title)
    return fig

def custom_draw_geometry_load_option(
    vis_list: List, width=1600, height=900, fov_step=-90,
):
    """
    Draw multiple open3d geometry objects in a single window.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    for geom in vis_list:
        vis.add_geometry(geom)
    # vis.get_render_option().load_from_json("./render_option.json")
    
    # NOTE: code for changing FoV currently does nothing, so manually do it in GUI with '['and ']'
    # ctr = vis.get_view_control()
    # print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    # ctr.change_field_of_view(step=fov_step)  # min fov is 5 deg, so a step of -90 will automatically step to the min fov
    # print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()

def create_spheres(
    centroids: ndarray,
    color=[0.1, 0.6, 0.2],
    radius=0.05,
    sphere_list: Optional[List] = None,
):
    """
    Create a list of spheres from a 2D numpy array.
    Centroids needs to be (Nx3).
    """
    vis_list = []
    for row_idx in range(centroids.shape[0]):
        c_pt = centroids[row_idx, :]
        if sphere_list is not None:
            mesh_sphere = sphere_list[row_idx].create_sphere(radius=radius)
        else:
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.compute_vertex_normals()
        if color is not None:
            mesh_sphere.paint_uniform_color(color)
        mesh_sphere.translate(c_pt)
        vis_list.append(mesh_sphere)
    return vis_list

def make_open3d_axes(axis_vectors=None, origin=None, scale=1.0):
    if origin is None:
        origin = np.zeros((1, 3))
    if axis_vectors is None:
        axis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    axis_vectors = axis_vectors * scale
    axis_points = origin + axis_vectors
    points = np.concatenate([origin, axis_points], axis=0)
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=int)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(lines)
    axes.colors = o3d.utility.Vector3dVector(colors)
    return axes

def random_non_red_colors(N):
    # Oversample to increase chance of getting enough valid colors
    M = int(N * 1.5)
    
    while True:
        colors = np.random.rand(M, 3)  # shape (M, 3), values in [0, 1]

        r = colors[:, 0]
        g = colors[:, 1]
        b = colors[:, 2]

        # Boolean mask to filter out red-dominant colors
        non_red_mask = ~((r > 0.7) & (r > g + 0.2) & (r > b + 0.2))

        filtered_colors = colors[non_red_mask]

        if len(filtered_colors) >= N:
            return filtered_colors[:N]

def isin_rowwise(arr1: ndarray, arr2: ndarray):
    """
    Row-wise comparison of two arrays. Returns the rows of arr1 that are in arr2.  
    Returns (N,) mask array
    """
    assert arr1.ndim == 2 and arr2.ndim == 2
    # Convert to structured arrays for row-wise comparison
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    return np.isin(arr1_view, arr2_view)[:,0]

def visualise_correspondences(
    anc_points_coarse: Union[Tensor, ndarray],
    pos_points_coarse: Union[Tensor, ndarray],
    anc_points_fine: Union[Tensor, ndarray],
    pos_points_fine: Union[Tensor, ndarray],
    node_corr_indices: ndarray,
    gt_node_corr_indices: ndarray,
    transform: ndarray,
    anc_point_to_node: Union[Tensor, ndarray, None] = None,
    pos_point_to_node: Union[Tensor, ndarray, None] = None,
    translate=[0, 0, 40],
):
    """
    Helper function for visualising keypoint correspondences.

    Args:
        anc_points_coarse: Source keypoints (nodes)
        pos_points_coarse: Target keypoints (nodes)
        anc_points_fine: Source points
        pos_points_fine: Target Points
        node_corr_indices: Node correspondences
        gt_node_corr_indices: Ground truth node correspondences
        transform: SE(3) transform from source to target
        anc_point_to_node: Index of points belonging to each source keypoint, used for colourising by patch instead of height
        pos_point_to_node: Index of points belonging to each target keypoint, used for colourising by patch instead of height
        translate: Translation applied to target for correspondence visualisation

    """
    # PC_SOURCE_COLOUR = [1, 0.7, 0.05]
    # PC_TARGET_COLOUR = [0, 0.629, 0.9]
    PC_SOURCE_COLOURMAP = 'viridis'
    PC_TARGET_COLOURMAP = 'gray'
    
    # KP_INLIER_COLOUR = [0.87, 0, 0.84]
    # KP_OUTLIER_COLOUR = [0.3, 0.3, 0.3]
    KP_UNUSED_COLOUR = [0.3, 0.3, 0.3]
    KP_INLIER_COLOUR = [0.0, 1.0, 0.0]
    KP_OUTLIER_COLOUR = [1.0, 0, 0]
    
    INLIER_CORRESPONDENCE_COLOUR = [0, 0.9, 0.1]
    OUTLIER_CORRESPONDENCE_COLOUR = [0.9, 0.1, 0]

    KP_RADIUS = 1.0

    # VOXEL_SIZE = 0.6

    anc_points_fine_o3d = make_open3d_point_cloud(anc_points_fine)
    pos_points_fine_o3d = make_open3d_point_cloud(pos_points_fine)
    anc_points_coarse_o3d = make_open3d_point_cloud(anc_points_coarse)
    pos_points_coarse_o3d = make_open3d_point_cloud(pos_points_coarse)

    # Align point clouds with gt transform
    anc_points_fine_o3d.transform(transform)
    anc_points_coarse_o3d.transform(transform)
    
    # Manually add offset to positive for ease of visualisation
    pos_points_fine_o3d.translate(translate)
    pos_points_coarse_o3d.translate(translate)

    # # Downsample point clouds for ease of visualisation
    # pc_source_o3d = pc_source_o3d.voxel_down_sample(VOXEL_SIZE)
    # pc_target_o3d = pc_target_o3d.voxel_down_sample(VOXEL_SIZE)

    # Filter inlier correspondences
    inlier_mask = isin_rowwise(node_corr_indices, gt_node_corr_indices)
    inlier_corr = node_corr_indices[inlier_mask]
    outlier_corr = node_corr_indices[~inlier_mask]

    # TODO: Filter unique correspondences, and perhaps only mutual ones? 
    #       (although I think mutual only applies to fine corr, not coarse, need
    #       to verify)
    #       ANSWER: No mutual filtering. Multiple coarse corr allowed, all are 
    #       then evaluated in LGR.
    
    # Create lineset between correspondences
    inlier_node_corr_lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        anc_points_coarse_o3d, pos_points_coarse_o3d, inlier_corr
    )
    outlier_node_corr_lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        anc_points_coarse_o3d, pos_points_coarse_o3d, outlier_corr
    )

    # Separate unused keypoints
    anc_points_coarse_full_ndx = np.arange(len(anc_points_coarse))
    pos_points_coarse_full_ndx = np.arange(len(pos_points_coarse))
    anc_points_coarse_unused_indices = np.setdiff1d(anc_points_coarse_full_ndx, node_corr_indices[:,0])
    pos_points_coarse_unused_indices = np.setdiff1d(pos_points_coarse_full_ndx, node_corr_indices[:,1])
    anc_points_coarse_unused = np.asarray(anc_points_coarse_o3d.points)[anc_points_coarse_unused_indices]
    pos_points_coarse_unused = np.asarray(pos_points_coarse_o3d.points)[pos_points_coarse_unused_indices]
    # Separate inliers and outliers
    anc_points_coarse_inliers = np.asarray(anc_points_coarse_o3d.points)[inlier_corr[:,0]]
    pos_points_coarse_inliers = np.asarray(pos_points_coarse_o3d.points)[inlier_corr[:,1]]
    anc_points_coarse_outliers = np.asarray(anc_points_coarse_o3d.points)[outlier_corr[:,0]]
    pos_points_coarse_outliers = np.asarray(pos_points_coarse_o3d.points)[outlier_corr[:,1]]

    # Plot spheres for the keypoints, and change colours appropriately
    anc_points_coarse_inliers_spheres_o3d = create_spheres(
        anc_points_coarse_inliers, color=KP_INLIER_COLOUR, radius=KP_RADIUS,
    )
    pos_points_coarse_inliers_spheres_o3d = create_spheres(
        pos_points_coarse_inliers, color=KP_INLIER_COLOUR, radius=KP_RADIUS,
    )
    anc_points_coarse_outliers_spheres_o3d = create_spheres(
        anc_points_coarse_outliers, color=KP_OUTLIER_COLOUR, radius=KP_RADIUS,
    )
    pos_points_coarse_outliers_spheres_o3d = create_spheres(
        pos_points_coarse_outliers, color=KP_OUTLIER_COLOUR, radius=KP_RADIUS,
    )
    anc_points_coarse_unused_spheres_o3d = create_spheres(
        anc_points_coarse_unused, color=KP_UNUSED_COLOUR, radius=KP_RADIUS,
    )
    pos_points_coarse_unused_spheres_o3d = create_spheres(
        pos_points_coarse_unused, color=KP_UNUSED_COLOUR, radius=KP_RADIUS,
    )

    # Set colours
    ## pc_source_o3d.paint_uniform_color(PC_SOURCE_COLOUR)
    ## pc_target_o3d.paint_uniform_color(PC_TARGET_COLOUR)
    inlier_node_corr_lineset.paint_uniform_color(INLIER_CORRESPONDENCE_COLOUR)
    outlier_node_corr_lineset.paint_uniform_color(OUTLIER_CORRESPONDENCE_COLOUR)

    # Colourise point clouds by z-coordinate
    if anc_point_to_node is not None:
        anc_node_colours = random_non_red_colors(anc_points_coarse.shape[0])
        anc_points_colours = anc_node_colours[anc_point_to_node]
    else:
        anc_points_colours = colourise_points_by_height(np.asarray(anc_points_fine_o3d.points), PC_SOURCE_COLOURMAP)
    if pos_point_to_node is not None:
        pos_node_colours = random_non_red_colors(pos_points_coarse.shape[0])
        pos_points_colours = pos_node_colours[pos_point_to_node]
    else:
        pos_points_colours = colourise_points_by_height(np.asarray(pos_points_fine_o3d.points), PC_TARGET_COLOURMAP)
    anc_points_fine_o3d.colors = o3d.utility.Vector3dVector(anc_points_colours)
    pos_points_fine_o3d.colors = o3d.utility.Vector3dVector(pos_points_colours)    

    # Add axes
    anc_axes = make_open3d_axes(scale=2.0)
    pos_axes = make_open3d_axes(origin=np.array([translate]), scale=2.0)

    # Draw all with Open3D
    vis_list = [anc_points_fine_o3d, pos_points_fine_o3d, inlier_node_corr_lineset, outlier_node_corr_lineset,
                *anc_points_coarse_unused_spheres_o3d, *pos_points_coarse_unused_spheres_o3d,
                *anc_points_coarse_outliers_spheres_o3d, *pos_points_coarse_outliers_spheres_o3d,
                *anc_points_coarse_inliers_spheres_o3d, *pos_points_coarse_inliers_spheres_o3d,
                anc_axes, pos_axes]
    # vis_list = [anc_points_coarse_o3d, pos_points_coarse_o3d, gt_inliers_o3d]
    custom_draw_geometry_load_option(vis_list)

def visualise_registration(
    anc_points_coarse: Union[Tensor, ndarray],
    pos_points_coarse: Union[Tensor, ndarray],
    anc_points_fine: Union[Tensor, ndarray],
    pos_points_fine: Union[Tensor, ndarray],
    node_corr_indices: ndarray,
    gt_node_corr_indices: ndarray,
    transform: ndarray,
):
    """
    WARNING: UNFINISHED
    Helper function for visualising registration.

    Args:
        anc_points_coarse: Source keypoints (nodes)
        pos_points_coarse: Target keypoints (nodes)
        anc_points_fine: Source points
        pos_points_fine: Target Points
        node_corr_indices: Node correspondences
        gt_node_corr_indices: Ground truth node correspondences
        transform: SE(3) transform from source to target

    """
    PC_SOURCE_COLOUR = [1, 0.7, 0.05]
    PC_TARGET_COLOUR = [0, 0.629, 0.9]
    # PC_SOURCE_COLOURMAP = 'viridis'
    # PC_TARGET_COLOURMAP = 'gray'
    
    # KP_INLIER_COLOUR = [0.87, 0, 0.84]
    # KP_OUTLIER_COLOUR = [0.3, 0.3, 0.3]
    KP_UNUSED_COLOUR = [0.3, 0.3, 0.3]
    KP_INLIER_COLOUR = [0.0, 1.0, 0.0]
    KP_OUTLIER_COLOUR = [1.0, 0, 0]
    
    INLIER_CORRESPONDENCE_COLOUR = [0, 0.9, 0.1]
    OUTLIER_CORRESPONDENCE_COLOUR = [0.9, 0.1, 0]

    KP_RADIUS = 1.0

    # VOXEL_SIZE = 0.6

    anc_points_fine_o3d = make_open3d_point_cloud(anc_points_fine)
    pos_points_fine_o3d = make_open3d_point_cloud(pos_points_fine)
    anc_points_coarse_o3d = make_open3d_point_cloud(anc_points_coarse)
    pos_points_coarse_o3d = make_open3d_point_cloud(pos_points_coarse)

    # Align point clouds with gt transform
    anc_points_fine_o3d.transform(transform)
    anc_points_coarse_o3d.transform(transform)
    
    # # Downsample point clouds for ease of visualisation
    # pc_source_o3d = pc_source_o3d.voxel_down_sample(VOXEL_SIZE)
    # pc_target_o3d = pc_target_o3d.voxel_down_sample(VOXEL_SIZE)

    # Filter inlier correspondences
    inlier_mask = isin_rowwise(node_corr_indices, gt_node_corr_indices)
    inlier_corr = node_corr_indices[inlier_mask]
    outlier_corr = node_corr_indices[~inlier_mask]

    # TODO: Filter unique correspondences, and perhaps only mutual ones? 
    #       (although I think mutual only applies to fine corr, not coarse, need
    #       to verify)
    #       ANSWER: No mutual filtering. Multiple coarse corr allowed, all are 
    #       then evaluated in LGR.
    
    # Create lineset between correspondences
    inlier_node_corr_lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        anc_points_coarse_o3d, pos_points_coarse_o3d, inlier_corr
    )
    outlier_node_corr_lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        anc_points_coarse_o3d, pos_points_coarse_o3d, outlier_corr
    )

    # Separate unused keypoints
    anc_points_coarse_full_ndx = np.arange(len(anc_points_coarse))
    pos_points_coarse_full_ndx = np.arange(len(pos_points_coarse))
    anc_points_coarse_unused_indices = np.setdiff1d(anc_points_coarse_full_ndx, node_corr_indices[:,0])
    pos_points_coarse_unused_indices = np.setdiff1d(pos_points_coarse_full_ndx, node_corr_indices[:,1])
    anc_points_coarse_unused = np.asarray(anc_points_coarse_o3d.points)[anc_points_coarse_unused_indices]
    pos_points_coarse_unused = np.asarray(pos_points_coarse_o3d.points)[pos_points_coarse_unused_indices]
    # Separate inliers and outliers
    anc_points_coarse_inliers = np.asarray(anc_points_coarse_o3d.points)[inlier_corr[:,0]]
    pos_points_coarse_inliers = np.asarray(pos_points_coarse_o3d.points)[inlier_corr[:,1]]
    anc_points_coarse_outliers = np.asarray(anc_points_coarse_o3d.points)[outlier_corr[:,0]]
    pos_points_coarse_outliers = np.asarray(pos_points_coarse_o3d.points)[outlier_corr[:,1]]

    # Plot spheres for the keypoints, and change colours appropriately
    anc_points_coarse_inliers_spheres_o3d = create_spheres(
        anc_points_coarse_inliers, color=KP_INLIER_COLOUR, radius=KP_RADIUS,
    )
    pos_points_coarse_inliers_spheres_o3d = create_spheres(
        pos_points_coarse_inliers, color=KP_INLIER_COLOUR, radius=KP_RADIUS,
    )
    anc_points_coarse_outliers_spheres_o3d = create_spheres(
        anc_points_coarse_outliers, color=KP_OUTLIER_COLOUR, radius=KP_RADIUS,
    )
    pos_points_coarse_outliers_spheres_o3d = create_spheres(
        pos_points_coarse_outliers, color=KP_OUTLIER_COLOUR, radius=KP_RADIUS,
    )
    anc_points_coarse_unused_spheres_o3d = create_spheres(
        anc_points_coarse_unused, color=KP_UNUSED_COLOUR, radius=KP_RADIUS,
    )
    pos_points_coarse_unused_spheres_o3d = create_spheres(
        pos_points_coarse_unused, color=KP_UNUSED_COLOUR, radius=KP_RADIUS,
    )

    # Set colours
    ## pc_source_o3d.paint_uniform_color(PC_SOURCE_COLOUR)
    ## pc_target_o3d.paint_uniform_color(PC_TARGET_COLOUR)
    inlier_node_corr_lineset.paint_uniform_color(INLIER_CORRESPONDENCE_COLOUR)
    outlier_node_corr_lineset.paint_uniform_color(OUTLIER_CORRESPONDENCE_COLOUR)

    anc_points_fine_o3d.paint_uniform_color(PC_SOURCE_COLOUR)
    pos_points_fine_o3d.paint_uniform_color(PC_TARGET_COLOUR)
    # anc_points_fine_o3d.colors = o3d.utility.Vector3dVector(PC_SOURCE_COLOUR)
    # pos_points_fine_o3d.colors = o3d.utility.Vector3dVector(PC_TARGET_COLOUR)    

    # Add axes
    anc_axes = make_open3d_axes(scale=2.0)

    # Draw all with Open3D
    vis_list = [anc_points_fine_o3d, pos_points_fine_o3d,
                # inlier_node_corr_lineset, outlier_node_corr_lineset,
                # *anc_points_coarse_unused_spheres_o3d, *pos_points_coarse_unused_spheres_o3d,
                # *anc_points_coarse_outliers_spheres_o3d, *pos_points_coarse_outliers_spheres_o3d,
                # *anc_points_coarse_inliers_spheres_o3d, *pos_points_coarse_inliers_spheres_o3d,
                anc_axes]
    # vis_list = [anc_points_coarse_o3d, pos_points_coarse_o3d, gt_inliers_o3d]
    custom_draw_geometry_load_option(vis_list)
