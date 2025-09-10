"""
Utility functions for visualising HOTFormerLoc and octrees.

Ethan Griffiths (Data61, Pullenvale)
"""
from typing import List, Optional, Union
import os

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
import umap
from scipy.spatial.transform import Rotation as R

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
    values: ndarray, colourmap_name='viridis', normalise=True,
) -> ndarray:
    """
    Colourise a point cloud based on scalar values using a matplotlib colourmap.
    
    Args:
        values: numpy array of shape (N, 1) containing scalar values to colourise each point by
        colourmap_name: name of the matplotlib colourmap to use (default: 'viridis')
        normalise: normalise values before applying colourmap
    
    Returns:
        colours: numpy array of shape (N, 3) containing RGB values in range [0, 1]
    """
    # Normalise values to range [0, 1]
    if normalise:
        if np.max(values) > np.min(values):
            values = (values - np.min(values)) / (np.max(values) - np.min(values))
        else:
            # Handle the case where all points have the same value
            values = np.zeros_like(values)
    
    # Get the colormap from matplotlib
    cmap = plt.get_cmap(colourmap_name)
    
    # Convert normalized values to colors
    colours = cmap(values)[:, :3]  # Only take RGB, discard alpha
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

def get_colours_by_embedding_similarity(data: ndarray, mode='tsne') -> ndarray:
    r"""
    TODO: Use this func for colourisation  
    Use t-SNE/umap to project high-dimension feats to rgb

    Args:
        data (ndarray): (N, C)
        mode (str): tsne or umap

    Returns:
        colors (ndarray): (N, 3)
    """
    mode = mode.lower()
    assert mode in ('tsne', 'umap')
    if mode == 'tsne':
        proj_func = TSNE(n_components=1, perplexity=40, n_iter=300, random_state=0)
    elif mode == 'umap':
        proj_func = umap.UMAP(n_components=1, n_neighbors=15, min_dist=0.1, metric='euclidean')
    proj_results = proj_func.fit_transform(data).reshape(-1)
    proj_min = np.min(proj_results)
    proj_max = np.max(proj_results)
    normalized_proj_results = (proj_results - proj_min) / (proj_max - proj_min)
    # colours = plt.cm.Spectral(normalized_proj_results)[:, :3]
    colours = plt.cm.rainbow(normalized_proj_results)[:, :3]
    return colours

def colourise_points_by_similarity(
    embeddings: ndarray, mode: str = 'tsne', return_explained_variance=False,
) -> Union[ndarray, List[ndarray]]:
    """
    Colourise a point cloud based on similarity of local features. Uses t-SNE or
    PCA to compute the colourisation.
    
    Args:
        embeddings: numpy array of shape (N, D) containing embeddings for each point
        mode: tSNE, UMAP, or PCA projection
        return_explained_variance: if using PCA, returns explained variance of each component
    
    Returns:
        colours: numpy array of shape (N, 3) containing RGB values in range [0, 1]
    """
    assert mode.lower() in ('tsne', 'pca', 'umap'), "mode must be 'tsne', 'umap', or 'pca'"
    assert embeddings.ndim == 2
    eps = 1e-8
    if mode.lower() == 'tsne':
        # NOTE: perplexity is not meant to be lower than n_samples, so may need to adjust this
        #       for shallow octree levels with only a handful of octants
        colours = get_colours_by_embedding_similarity(embeddings, 'tsne')
    elif mode.lower() == 'umap':
        colours = get_colours_by_embedding_similarity(embeddings, 'umap')
    elif mode.lower() == 'pca':
        pca = PCA(n_components=3)
        colours = pca.fit_transform(embeddings)
        pca_explained_variance = pca.explained_variance_ratio_

    # Normalize to [0, 1]
    colours = (colours - colours.min(0)) / (colours.max(0) - colours.min(0) + eps)

    if mode.lower() == 'pca' and return_explained_variance:
        return colours, pca_explained_variance
    return colours

def create_heatmap(
    values: Tensor,
    ticklabels: Optional[List] = None,
    min_value: Optional[float] = None,
    title: Optional[str] = None,
) -> plt.Figure:
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

def set_view_control(vis: o3d.visualization.Visualizer, fov_step=-90, zoom=0.55, angle=-380.0):
    """
    Set default viewpoint for open3d Visualizer (orthographic view, 35 deg angle).
    """
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=fov_step)  # make view orthographic
    ctr.rotate(0.0, angle)  # set camera angle (~38deg)
    ctr.set_zoom(zoom)

def set_initial_rotation(
    vis: o3d.visualization.Visualizer,
    vis_list: List[o3d.geometry.Geometry],
    rotation: float = 45.0,
):
    """
    Set initial z rotation of geometries
    """
    # Get centre for rotation from first PointCloud object
    # NOTE: use this instead of origin to ensure viewing window stays more consistent
    rot_centre = np.zeros((3, 1), dtype=np.float64)  # origin
    for geom in vis_list:
        if not isinstance(geom, o3d.geometry.PointCloud):
            continue
        pcd = np.asarray(vis_list[0].points)
        bbmin = pcd.min(axis=0)
        bbmax = pcd.max(axis=0)
        rot_centre = (bbmin + bbmax) * 0.5
        break
    rotation_matrix = R.from_euler('z', rotation, degrees=True).as_matrix()

    # Set rotation
    for geom in vis_list:
        geom.rotate(rotation_matrix, rot_centre)
        vis.update_geometry(geom)

def custom_draw_geometry_with_z_rotation(
    vis_list: List[o3d.geometry.Geometry],
    rot_step=1.0,
    width=1600,
    height=900,
    fov_step=-90,
    zoom=0.55,
    angle=-380.0,
    save_dir: Optional[str] = None,
):
    """
    Animate rotation around z axis (rot_step in degrees)
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    view_json_path = os.path.join(os.path.split(__file__)[0], 'o3d_viz.json')
    custom_draw_geometry_with_z_rotation.index = 0
    num_steps = 360 // rot_step
    # Assume first geometry item is point cloud - get centre for rotation
    pcd = np.asarray(vis_list[0].points)
    bbmin = pcd.min(axis=0)
    bbmax = pcd.max(axis=0)
    rot_centre = (bbmin + bbmax) * 0.5
    # rot_centre = np.zeros((3, 1), dtype=np.float64)  # origin

    def rotate_geometry(vis: o3d.visualization.Visualizer):
        glb = custom_draw_geometry_with_z_rotation
        if glb.index == 0:
            set_view_control(vis, fov_step, zoom, angle)
            set_initial_rotation(vis, vis_list)
        rot = R.from_euler('z', rot_step, degrees=True).as_matrix()
        for geom in vis_list:
            geom.rotate(rot, rot_centre)
            vis.update_geometry(geom)
        # TODO: save img to file
        if glb.index < num_steps:
            if save_dir is not None:
                vis.capture_screen_image(os.path.join(save_dir, f'{glb.index:05d}.png'), True)
                # image = vis.capture_screen_float_buffer(False)
                # plt.imsave(os.path.join(frame_dir, f'{glb.index:05d}.png'),
                #            np.asarray(image),
                #            dpi=1)
        else:  # quit after saving animation
            vis.destroy_window()
        glb.index += 1
        return True
        
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    for geom in vis_list:
        vis.add_geometry(geom)
    vis.get_render_option().load_from_json(view_json_path)
    vis.register_animation_callback(rotate_geometry)
    vis.run()
    vis.destroy_window()

def custom_draw_geometry_with_rotation(vis_list: List[o3d.geometry.Geometry]):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False
    o3d.visualization.draw_geometries_with_animation_callback(vis_list,
                                                              rotate_view)

def custom_draw_geometry_load_option(
    vis_list: List[o3d.geometry.Geometry],
    width=1600,
    height=900,
    fov_step=-90,
    zoom=0.55,
    angle=-380.0,
    save_dir: Optional[str] = None,
    filename: str = 'frame',
    non_interactive=False,
):
    """
    Draw multiple open3d geometry objects in a single window.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    custom_draw_geometry_load_option.index = 0
    def viz_callback(vis: o3d.visualization.Visualizer):
        glb = custom_draw_geometry_load_option
        if glb.index == 0:
            set_view_control(vis, fov_step, zoom, angle)
            set_initial_rotation(vis, vis_list)
            if save_dir is not None:
                vis.capture_screen_image(os.path.join(save_dir, f'{filename}.png'), True)
            if non_interactive:
                vis.destroy_window()
        glb.index += 1
        return False
    view_json_path = os.path.join(os.path.split(__file__)[0], 'o3d_viz.json')
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    for geom in vis_list:
        vis.add_geometry(geom)
    vis.get_render_option().load_from_json(view_json_path)
    vis.register_animation_callback(viz_callback)
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
        non_red_mask = ~((r > 0.8) & (r > g + 0.3) & (r > b + 0.3))

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

def visualise_coarse_correspondences(
    anc_points_coarse: Union[Tensor, ndarray],
    pos_points_coarse: Union[Tensor, ndarray],
    anc_points_fine: Union[Tensor, ndarray],
    pos_points_fine: Union[Tensor, ndarray],
    node_corr_indices: ndarray,
    gt_node_corr_indices: ndarray,
    transform: ndarray,
    anc_point_to_node: Union[Tensor, ndarray, None] = None,
    pos_point_to_node: Union[Tensor, ndarray, None] = None,
    anc_feats_coarse: Optional[ndarray] = None,
    pos_feats_coarse: Optional[ndarray] = None,
    translate=[0, 0, 40],
    zoom=0.55,
    plot_coarse=False,
    coarse_colourmode: str = 'patch',
    save_dir: Optional[str] = None,
    disable_animation=False,
    non_interactive=False,
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
        anc_feats_coarse: Source feats (nodes)
        pos_feats_coarse: Target feats (nodes)
        translate: Translation applied to target for correspondence visualisation
        plot_coarse: Plot coarse points (patch centroids)
        coarse_colourmode: Mode for colourising patches ('height', 'patch', 'tsne', 'umap')
        save_dir: Directory to save plots
        disable_animation: Disables animation
    """
    coarse_colourmode = coarse_colourmode.lower()
    assert coarse_colourmode in ('height', 'patch', 'tsne', 'umap')
    if coarse_colourmode != 'height':
        assert pos_point_to_node is not None and anc_point_to_node is not None
    if coarse_colourmode in ('tsne', 'umap'):
        assert anc_feats_coarse is not None and pos_feats_coarse is not None
    # PC_SOURCE_COLOUR = [1, 0.7, 0.05]
    # PC_TARGET_COLOUR = [0, 0.629, 0.9]
    PC_SOURCE_COLOURMAP = 'viridis'
    PC_TARGET_COLOURMAP = 'gray'
    
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
    if plot_coarse:
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

    # Colourise point clouds
    if coarse_colourmode in ('tsne', 'umap'):
        combined_feats_coarse = np.concatenate([anc_feats_coarse, pos_feats_coarse], axis=0)
        combined_node_colours = colourise_points_by_similarity(combined_feats_coarse, mode=coarse_colourmode)
        anc_node_colours, pos_node_colours = np.split(combined_node_colours, [anc_feats_coarse.shape[0]], axis=0)
    elif coarse_colourmode == 'patch':
        anc_node_colours = random_non_red_colors(anc_points_coarse.shape[0])
        pos_node_colours = random_non_red_colors(pos_points_coarse.shape[0])
    if coarse_colourmode != 'height':
        anc_points_colours = anc_node_colours[anc_point_to_node]
        pos_points_colours = pos_node_colours[pos_point_to_node]
    else:
        anc_points_colours = colourise_points_by_height(np.asarray(anc_points_fine_o3d.points), PC_SOURCE_COLOURMAP)
        pos_points_colours = colourise_points_by_height(np.asarray(pos_points_fine_o3d.points), PC_TARGET_COLOURMAP)
    anc_points_fine_o3d.colors = o3d.utility.Vector3dVector(anc_points_colours)
    pos_points_fine_o3d.colors = o3d.utility.Vector3dVector(pos_points_colours)    

    # # Add axes
    # anc_axes = make_open3d_axes(scale=2.0)
    # pos_axes = make_open3d_axes(origin=np.array([translate]), scale=2.0)

    # Draw all with Open3D
    vis_list = [anc_points_fine_o3d, pos_points_fine_o3d,
                inlier_node_corr_lineset, outlier_node_corr_lineset,]
                # anc_axes, pos_axes]
    if plot_coarse:
        vis_list.extend([
            *anc_points_coarse_unused_spheres_o3d, *pos_points_coarse_unused_spheres_o3d,
            *anc_points_coarse_outliers_spheres_o3d, *pos_points_coarse_outliers_spheres_o3d,
            *anc_points_coarse_inliers_spheres_o3d, *pos_points_coarse_inliers_spheres_o3d,
          ])
    # vis_list = [anc_points_coarse_o3d, pos_points_coarse_o3d, gt_inliers_o3d]

    if disable_animation:
        custom_draw_geometry_load_option(  # static
            vis_list,
            save_dir=save_dir,
            filename=f'coarse_corr_{coarse_colourmode}',
            non_interactive=non_interactive,
            zoom=zoom,
        )
    else:
        if save_dir is not None:
            save_dir = os.path.join(save_dir, f'coarse_corr_{coarse_colourmode}_frames')
        custom_draw_geometry_with_z_rotation(
            vis_list,
            save_dir=save_dir,
            zoom=zoom,
        )  # with rotation


def visualise_fine_correspondences(
    anc_points_fine: Union[Tensor, ndarray],
    pos_points_fine: Union[Tensor, ndarray],
    anc_corr_points: Union[Tensor, ndarray],
    pos_corr_points: Union[Tensor, ndarray],
    corr_scores: ndarray,
    transform: ndarray,
    score_threshold: float = 0.0,
    anc_feats_fine: Optional[ndarray] = None,
    pos_feats_fine: Optional[ndarray] = None,
    translate=[0, 0, 40],
    zoom=0.55,
    colourmode: str = 'umap',
    save_dir: Optional[str] = None,
    disable_animation=False,
    non_interactive=False,
):
    """
    Helper function for visualising fine correspondences.

    Args:
        anc_points_fine: Source points
        pos_points_fine: Target points
        anc_corr_points: Source point correspondences
        pos_corr_points: Target point correspondences
        corr_scores: Correspondence scores
        transform: SE(3) transform from source to target
        score_threshold: threshold to filter correspondences with score less than threshold
        anc_feats_fine: Source feats
        pos_feats_fine: Target feats
        translate: Translation applied to target for correspondence visualisation
        plot_coarse: Plot coarse points (patch centroids)
        colourmode: Mode for colourising patches ('height', 'tsne', 'umap')
        save_dir: Directory to save plots
        disable_animation: Disables animation
    """
    colourmode = colourmode.lower()
    assert colourmode in ('height', 'tsne', 'umap')
    if colourmode in ('tsne', 'umap'):
        assert anc_feats_fine is not None and pos_feats_fine is not None

    PC_SOURCE_COLOURMAP = 'viridis'
    PC_TARGET_COLOURMAP = 'gray'
    PC_SOURCE_COLOUR = [1, 0.7, 0.05]
    PC_TARGET_COLOUR = [0, 0.629, 0.9]
    # CORRESPONDENCE_COLOURMAP = 'coolwarm'
    CORRESPONDENCE_COLOURMAP = 'RdYlGn'

    # VOXEL_SIZE = 0.6

    # Filter low confidence correspondences
    score_mask = corr_scores >= score_threshold
    corr_scores = corr_scores[score_mask]
    anc_corr_points = anc_corr_points[score_mask]
    pos_corr_points = pos_corr_points[score_mask]

    anc_points_fine_o3d = make_open3d_point_cloud(anc_points_fine)
    pos_points_fine_o3d = make_open3d_point_cloud(pos_points_fine)
    anc_corr_points_o3d = make_open3d_point_cloud(anc_corr_points)
    pos_corr_points_o3d = make_open3d_point_cloud(pos_corr_points)

    # Align point clouds with gt transform
    anc_points_fine_o3d.transform(transform)
    anc_corr_points_o3d.transform(transform)
    
    # Manually add offset to positive for ease of visualisation
    pos_points_fine_o3d.translate(translate)
    pos_corr_points_o3d.translate(translate)

    # # Downsample point clouds for ease of visualisation
    # pc_source_o3d = pc_source_o3d.voxel_down_sample(VOXEL_SIZE)
    # pc_target_o3d = pc_target_o3d.voxel_down_sample(VOXEL_SIZE)

    # Create lineset between correspondences
    assert len(anc_corr_points) == len(pos_corr_points)
    corr_indices = [(i, i) for i in range(len(anc_corr_points))]
    corr_points_lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        anc_corr_points_o3d, pos_corr_points_o3d, corr_indices,
    )

    # Set colours
    # inlier_node_corr_lineset.paint_uniform_color(INLIER_CORRESPONDENCE_COLOUR)
    # outlier_node_corr_lineset.paint_uniform_color(OUTLIER_CORRESPONDENCE_COLOUR)

    # Colourise correspondences by scores
    corr_points_lineset.colors = o3d.utility.Vector3dVector(
        colourise_points(corr_scores, colourmap_name=CORRESPONDENCE_COLOURMAP, normalise=False)
    )

    # Colourise point clouds
    if colourmode in ('tsne', 'umap'):
        combined_feats_fine = np.concatenate([anc_feats_fine, pos_feats_fine], axis=0)
        combined_points_fine_colours = colourise_points_by_similarity(combined_feats_fine, mode=colourmode)
        anc_points_fine_colours, pos_points_fine_colours = np.split(
            combined_points_fine_colours, [anc_feats_fine.shape[0]], axis=0,
        )
    else:
        anc_points_fine_colours = colourise_points_by_height(np.asarray(anc_points_fine_o3d.points), PC_SOURCE_COLOURMAP)
        pos_points_fine_colours = colourise_points_by_height(np.asarray(pos_points_fine_o3d.points), PC_TARGET_COLOURMAP)
    anc_points_fine_o3d.colors = o3d.utility.Vector3dVector(anc_points_fine_colours)
    pos_points_fine_o3d.colors = o3d.utility.Vector3dVector(pos_points_fine_colours)    

    # # Solid colours
    # anc_points_fine_o3d.paint_uniform_color(PC_SOURCE_COLOUR)
    # pos_points_fine_o3d.paint_uniform_color(PC_TARGET_COLOUR)
    anc_corr_points_o3d.paint_uniform_color(PC_SOURCE_COLOUR)
    pos_corr_points_o3d.paint_uniform_color(PC_TARGET_COLOUR)

    # Draw all with Open3D
    vis_list = [
        anc_points_fine_o3d, pos_points_fine_o3d,
        # anc_corr_points_o3d, pos_corr_points_o3d,
        corr_points_lineset,
    ]

    if disable_animation:
        custom_draw_geometry_load_option(  # static
            vis_list,
            save_dir=save_dir,
            filename=f'fine_corr_{colourmode}',
            non_interactive=non_interactive,
            zoom=zoom,
        )
    else:
        if save_dir is not None:
            save_dir = os.path.join(save_dir, f'fine_corr_{colourmode}_frames')
        custom_draw_geometry_with_z_rotation(
            vis_list,
            save_dir=save_dir,
            zoom=zoom,
        )  # with rotation


def visualise_similarity(
    anc_points_fine: Union[Tensor, ndarray],
    pos_points_fine: Union[Tensor, ndarray],
    transform: ndarray,
    anc_point_to_node: Union[Tensor, ndarray],
    pos_point_to_node: Union[Tensor, ndarray],
    anc_feats_coarse: ndarray,
    pos_feats_coarse: ndarray,
    translate=[0, 0, 40],
    zoom=0.55,
    coarse_colourmode: str = 'patch',
    save_dir: Optional[str] = None,
    disable_animation=False,
    non_interactive=False,
):
    """
    Helper function for visualising similarity of point cloud features

    Args:
        anc_points_fine: Source points
        pos_points_fine: Target Points
        transform: SE(3) transform from source to target
        anc_point_to_node: Index of points belonging to each source keypoint, used for colourising points by their patch feature
        pos_point_to_node: Index of points belonging to each target keypoint, used for colourising points by their patch feature
        anc_feats_coarse: Source feats (nodes)
        pos_feats_coarse: Target feats (nodes)
        translate: Translation applied to target for correspondence visualisation
        coarse_colourmode: Mode for colourising patches ('tsne', 'umap')
        save_dir: Directory to save plots
        disable_animation: Disables animation
    """
    coarse_colourmode = coarse_colourmode.lower()
    assert coarse_colourmode in ('tsne', 'umap')
    # VOXEL_SIZE = 0.6

    anc_points_fine_o3d = make_open3d_point_cloud(anc_points_fine)
    pos_points_fine_o3d = make_open3d_point_cloud(pos_points_fine)

    # Align point clouds with gt transform
    anc_points_fine_o3d.transform(transform)
    
    # Manually add offset to positive for ease of visualisation
    pos_points_fine_o3d.translate(translate)

    # # Downsample point clouds for ease of visualisation
    # pc_source_o3d = pc_source_o3d.voxel_down_sample(VOXEL_SIZE)
    # pc_target_o3d = pc_target_o3d.voxel_down_sample(VOXEL_SIZE)

    # Colourise point clouds
    if coarse_colourmode in ('tsne', 'umap'):
        combined_feats_coarse = np.concatenate([anc_feats_coarse, pos_feats_coarse], axis=0)
        combined_node_colours = colourise_points_by_similarity(combined_feats_coarse, mode=coarse_colourmode)
        anc_node_colours, pos_node_colours = np.split(combined_node_colours, [anc_feats_coarse.shape[0]], axis=0)
    anc_points_colours = anc_node_colours[anc_point_to_node]
    pos_points_colours = pos_node_colours[pos_point_to_node]
    anc_points_fine_o3d.colors = o3d.utility.Vector3dVector(anc_points_colours)
    pos_points_fine_o3d.colors = o3d.utility.Vector3dVector(pos_points_colours)    

    # Draw all with Open3D
    vis_list = [anc_points_fine_o3d, pos_points_fine_o3d,]

    if disable_animation:
        custom_draw_geometry_load_option(  # static
            vis_list,
            save_dir=save_dir,
            filename=f'coarse_sim_{coarse_colourmode}',
            non_interactive=non_interactive,
            zoom=zoom,
        )
    else:
        if save_dir is not None:
            save_dir = os.path.join(save_dir, f'coarse_sim_{coarse_colourmode}_frames')
        custom_draw_geometry_with_z_rotation(
            vis_list,
            save_dir=save_dir,
            zoom=zoom
        )  # with rotation


def visualise_registration(
    anc_points_fine: Union[Tensor, ndarray],
    pos_points_fine: Union[Tensor, ndarray],
    transform: ndarray,
    zoom=0.55,
    save_dir: Optional[str] = None,
    filename: str = 'registration',
    disable_animation=False,
    non_interactive=False,
):
    """
    Helper function for visualising registration.

    Args:
        anc_points_fine: Source points
        pos_points_fine: Target Points
        transform: SE(3) transform from source to target
        save_dir: Directory to save plots
        disable_animation: Disables animation
    """
    PC_SOURCE_COLOUR = [1, 0.7, 0.05]
    PC_TARGET_COLOUR = [0, 0.629, 0.9]

    # VOXEL_SIZE = 0.6

    anc_points_fine_o3d = make_open3d_point_cloud(anc_points_fine)
    pos_points_fine_o3d = make_open3d_point_cloud(pos_points_fine)

    # Align point clouds with gt transform
    anc_points_fine_o3d.transform(transform)
    
    # # Downsample point clouds for ease of visualisation
    # pc_source_o3d = pc_source_o3d.voxel_down_sample(VOXEL_SIZE)
    # pc_target_o3d = pc_target_o3d.voxel_down_sample(VOXEL_SIZE)

    # Set colours
    ## pc_source_o3d.paint_uniform_color(PC_SOURCE_COLOUR)
    ## pc_target_o3d.paint_uniform_color(PC_TARGET_COLOUR)

    anc_points_fine_o3d.paint_uniform_color(PC_SOURCE_COLOUR)
    pos_points_fine_o3d.paint_uniform_color(PC_TARGET_COLOUR)
    # anc_points_fine_o3d.colors = o3d.utility.Vector3dVector(PC_SOURCE_COLOUR)
    # pos_points_fine_o3d.colors = o3d.utility.Vector3dVector(PC_TARGET_COLOUR)    

    # Add axes
    # anc_axes = make_open3d_axes(scale=2.0)

    # Draw all with Open3D
    vis_list = [anc_points_fine_o3d, pos_points_fine_o3d,]
                # anc_axes]
    if disable_animation:
        custom_draw_geometry_load_option(
            vis_list,
            save_dir=save_dir,
            filename=filename,
            non_interactive=non_interactive,
            zoom=zoom,
        )
    else:
        if save_dir is not None:
            save_dir = os.path.join(save_dir, f'{filename}_frames')
        custom_draw_geometry_with_z_rotation(
            vis_list,
            save_dir=save_dir,
            zoom=zoom,
        )  # with rotation

def visualise_LGR_initial_registration(
    anc_corr_points: Union[Tensor, ndarray],
    pos_corr_points: Union[Tensor, ndarray],
    corr_scores: ndarray,
    transform: ndarray,
    translate=[0, 0, 0],
    zoom=0.55,
    angle=-380.0,
    save_dir: Optional[str] = None,
    disable_animation=False,
    non_interactive=False,
):
    """
    Helper function for visualising fine correspondences.

    Args:
        anc_corr_points: Source point correspondences
        pos_corr_points: Target point correspondences
        corr_scores: Correspondence scores
        transform: SE(3) transform from source to target
        save_dir: Directory to save plots
        disable_animation: Disables animation
    """
    PC_SOURCE_COLOUR = [1, 0.7, 0.05]
    PC_TARGET_COLOUR = [0, 0.629, 0.9]
    CORRESPONDENCE_COLOUR = [0, 1.0, 0]
    CORRESPONDENCE_COLOURMAP = 'RdYlGn'

    # VOXEL_SIZE = 0.6
    anc_corr_points_o3d = make_open3d_point_cloud(anc_corr_points)
    pos_corr_points_o3d = make_open3d_point_cloud(pos_corr_points)

    # Align point clouds with transform
    anc_corr_points_o3d.transform(transform)

    # Manually add offset to positive for ease of visualisation
    pos_corr_points_o3d.translate(translate)

    # Create lineset between correspondences
    assert len(anc_corr_points) == len(pos_corr_points)
    corr_indices = [(i, i) for i in range(len(anc_corr_points))]
    corr_points_lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        anc_corr_points_o3d, pos_corr_points_o3d, corr_indices,
    )

    # Set colours
    # corr_points_lineset.paint_uniform_color(CORRESPONDENCE_COLOUR)

    # Colourise correspondences by scores
    corr_points_lineset.colors = o3d.utility.Vector3dVector(
        colourise_points(corr_scores, colourmap_name=CORRESPONDENCE_COLOURMAP, normalise=True)
    )

    # # Solid colours
    anc_corr_points_o3d.paint_uniform_color(PC_SOURCE_COLOUR)
    pos_corr_points_o3d.paint_uniform_color(PC_TARGET_COLOUR)

    # Draw all with Open3D
    vis_list = [
        anc_corr_points_o3d, pos_corr_points_o3d,
        corr_points_lineset,
    ]

    if disable_animation:
        custom_draw_geometry_load_option(
            vis_list,
            save_dir=save_dir,
            filename='registration_LGR_initial_corr',
            non_interactive=non_interactive,
            zoom=zoom,
            angle=angle,
        )
    else:
        if save_dir is not None:
            save_dir = os.path.join(save_dir, 'registration_LGR_initial_corr_frames')
        custom_draw_geometry_with_z_rotation(
            vis_list,
            save_dir=save_dir,
            zoom=zoom,
            angle=angle,
        )  # with rotation

def visualise_relay_tokens(
    anc_points: Union[Tensor, ndarray, None],
    pos_points: Union[Tensor, ndarray, None],
    anc_rt_centroids_dict: dict,
    pos_rt_centroids_dict: dict,
    rt_centroid_correspondence_dict: dict,
    transform: ndarray,
    translate=[0, 0, 40],
    zoom=0.55,
    colourmode: str = 'patch',
    show_points=True,
    save_dir: Optional[str] = None,
    disable_animation=False,
    non_interactive=False,
):
    """
    Helper function for visualising relay tokens

    Args:
        TODO:
        anc_points_coarse: Source keypoints (nodes)
        pos_points_coarse: Target keypoints (nodes)
        anc_points_fine: Source points
        pos_points_fine: Target Points
        node_corr_indices: Node correspondences
        gt_node_corr_indices: Ground truth node correspondences
        transform: SE(3) transform from source to target
        anc_point_to_node: Index of points belonging to each source keypoint, used for colourising by patch instead of height
        pos_point_to_node: Index of points belonging to each target keypoint, used for colourising by patch instead of height
        anc_feats_coarse: Source feats (nodes)
        pos_feats_coarse: Target feats (nodes)
        translate: Translation applied to target for correspondence visualisation
        plot_coarse: Plot coarse points (patch centroids)
        coarse_colourmode: Mode for colourising patches ('height', 'patch', 'tsne', 'umap')
        save_dir: Directory to save plots
        disable_animation: Disables animation
    """
    colourmode = colourmode.lower()
    assert colourmode in ('height', 'patch')
    ANC_PC_COLOUR = np.array([1, 0.7, 0.05])
    POS_PC_COLOUR = np.array([0, 0.629, 0.9])
    # PC_SOURCE_COLOURMAP = 'viridis'
    # PC_TARGET_COLOURMAP = 'gray'
    
    # ANC_RT_COLOUR = [1.0, 0.0, 0.0]
    # POS_RT_COLOUR = [0.0, 1.0, 0.0]
    ANC_RT_COLOUR = np.array([1, 0.7, 0.05])
    POS_RT_COLOUR = np.array([0, 0.629, 0.9])
    CORRESPONDENCE_COLOUR = np.array([0.0, 1.0, 0.0])

    # KP_RADIUS = 1.0
    RT_BASE_RADIUS = 1.0

    # VOXEL_SIZE = 0.6

    anc_points_o3d = make_open3d_point_cloud(anc_points)
    pos_points_o3d = make_open3d_point_cloud(pos_points)

    # # Downsample point clouds for ease of visualisation
    # pc_source_o3d = pc_source_o3d.voxel_down_sample(VOXEL_SIZE)
    # pc_target_o3d = pc_target_o3d.voxel_down_sample(VOXEL_SIZE)

    # Plot spheres for relay tokens, and change colours appropriately
    anc_rt_spheres_o3d = []
    pos_rt_spheres_o3d = []
    for ii, depth_j in enumerate(anc_rt_centroids_dict.keys()):
        rt_radius_depth_j = RT_BASE_RADIUS + (RT_BASE_RADIUS / 2) * ii
        anc_rt_spheres_o3d.extend(
            create_spheres(anc_rt_centroids_dict[depth_j], color=ANC_RT_COLOUR / (ii+1), radius=rt_radius_depth_j)
        )
        pos_rt_spheres_o3d.extend(
            create_spheres(pos_rt_centroids_dict[depth_j], color=POS_RT_COLOUR / (ii+1), radius=rt_radius_depth_j)
        )

    # Align point clouds with gt transform
    anc_points_o3d.transform(transform)
    for anc_rt_sphere in anc_rt_spheres_o3d:
        anc_rt_sphere.transform(transform)
    
    # Manually add offset to positive for ease of visualisation
    pos_points_o3d.translate(translate)
    for pos_rt_sphere in pos_rt_spheres_o3d:
        pos_rt_sphere.translate(translate)

    # Create lineset between correspondences
    rt_centroid_correspondences_o3d = []
    for ii, depth_j in enumerate(rt_centroid_correspondence_dict.keys()):
        rt_centroid_correspondences_o3d.append(
            o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                anc_points_o3d, pos_points_o3d, rt_centroid_correspondence_dict[depth_j]
            )
        )
        rt_centroid_correspondences_o3d[ii].paint_uniform_color(CORRESPONDENCE_COLOUR / (ii+1))  # scale colour with depth

    # Set colours
    anc_points_o3d.paint_uniform_color(ANC_PC_COLOUR)
    pos_points_o3d.paint_uniform_color(POS_PC_COLOUR)

    # # Colourise point clouds
    # if coarse_colourmode in ('tsne', 'umap'):
    #     combined_feats_coarse = np.concatenate([anc_feats_coarse, pos_feats_coarse], axis=0)
    #     combined_node_colours = colourise_points_by_similarity(combined_feats_coarse, mode=coarse_colourmode)
    #     anc_node_colours, pos_node_colours = np.split(combined_node_colours, [anc_feats_coarse.shape[0]], axis=0)
    # elif coarse_colourmode == 'patch':
    #     anc_node_colours = random_non_red_colors(anc_points_coarse.shape[0])
    #     pos_node_colours = random_non_red_colors(pos_points_coarse.shape[0])
    # if coarse_colourmode != 'height':
    #     anc_points_colours = anc_node_colours[anc_point_to_node]
    #     pos_points_colours = pos_node_colours[pos_point_to_node]
    # else:
    #     anc_points_colours = colourise_points_by_height(np.asarray(anc_points_o3d.points), PC_SOURCE_COLOURMAP)
    #     pos_points_colours = colourise_points_by_height(np.asarray(pos_points_o3d.points), PC_TARGET_COLOURMAP)
    # anc_points_o3d.colors = o3d.utility.Vector3dVector(anc_points_colours)
    # pos_points_o3d.colors = o3d.utility.Vector3dVector(pos_points_colours)    

    # Draw all with Open3D
    vis_list = []
    if show_points:
        vis_list.extend([anc_points_o3d, pos_points_o3d,])
    vis_list.extend([
        *anc_rt_spheres_o3d, *pos_rt_spheres_o3d,
        # *rt_centroid_correspondences_o3d,  # Currently causes issues
        ])

    if disable_animation:
        custom_draw_geometry_load_option(  # static
            vis_list,
            save_dir=save_dir,
            filename=f'relay_tokens_{colourmode}',
            non_interactive=non_interactive,
            zoom=zoom,
        )
    else:
        if save_dir is not None:
            save_dir = os.path.join(save_dir, f'relay_tokens_{colourmode}_frames')
        custom_draw_geometry_with_z_rotation(
            vis_list,
            save_dir=save_dir,
            zoom=zoom,
        )  # with rotation

