"""
Utility functions for visualising HOTFormerLoc and octrees.

Ethan Griffiths (Data61, Pullenvale)
"""
from typing import List
import numpy as np
import torch
from torch import Tensor
import ocnn
import matplotlib.pyplot as plt
from dataset.coordinate_utils import CylindricalCoordinates
from misc.utils import TrainingParams, rescale_octree_points
from models.octree import OctreeT

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

def get_octree_points_and_windows(query_octree: OctreeT, depth: int, params: TrainingParams):
    """
    For a given octree depth, returns the octree points, octree points reshaped
    to windows, and a tensor containing the idx of each window. 
    """
    key = query_octree.key(depth, nempty=True)
    x, y, z, _ = ocnn.octree.key2xyz(key, depth)
    xyz = torch.stack([x, y, z], dim=1)
    # Convert octree point coords to original scale
    points_octree = rescale_octree_points(xyz, depth)
    # Undo cylindrical projection
    if params.model_params.coordinates == 'cylindrical':
        coord_converter = CylindricalCoordinates(use_octree=True)
        points_octree = coord_converter.undo_conversion(points_octree)        
    # Create window partitions and get idx
    points_octree_windows = query_octree.data_to_windows(points_octree, depth, False)
    windows_idx = torch.zeros(points_octree_windows.shape[:-1],
                              dtype=torch.int32)
    num_windows = len(windows_idx)
    # generate idx of windows
    idx_values = torch.arange(num_windows, dtype=torch.int32).unsqueeze(-1)
    windows_idx += idx_values
    # Reverse patch operation and remove padding
    windows_idx = windows_idx.reshape(-1)
    windows_idx = query_octree.patch_reverse(windows_idx, depth)
    return points_octree, points_octree_windows, windows_idx

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

def colourise_points_by_height(
    points: np.ndarray, colourmap_name='viridis',
) -> np.ndarray:
    """
    Colourise a point cloud based on z values using a matplotlib colourmap.
    
    Args:
        points: numpy array of shape (N, 3) containing 3D points
        colourmap_name: name of the matplotlib colourmap to use (default: 'viridis')
    
    Returns:
        colours: numpy array of shape (N, 3) containing RGB values in range [0, 1]
    """
    # Extract z-values (height)
    z_values = points[:, 2]
    
    # Normalize z-values to range [0, 1]
    if np.max(z_values) > np.min(z_values):
        z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
    else:
        # Handle the case where all points have the same z-value
        z_normalized = np.zeros_like(z_values)
    
    # Get the colormap from matplotlib
    cmap = plt.get_cmap(colourmap_name)
    
    # Convert normalized values to colors
    colours = cmap(z_normalized)[:, :3]  # Only take RGB, discard alpha
    return colours