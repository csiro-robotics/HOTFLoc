"""
Visualise hierarchical relay token attention maps.

Ethan Griffiths (QUT & CSIRO Data61).
"""
from typing import Dict, List, Optional
import numpy as np
import pickle
import os
import argparse
import torch
from torch import Tensor
import torch.nn.functional as F
import MinkowskiEngine as ME
import random
from tqdm import tqdm
import ocnn
from ocnn.octree import Octree, Points
from models.octree import OctreeT, get_octree_points_and_windows, rescale_octree_points
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from models.model_factory import model_factory
from misc.utils import TrainingParams
from misc.torch_utils import set_seed
from dataset.pointnetvlad.pnv_raw import PNVPointCloudLoader
from dataset.AboveUnder.AboveUnder_raw import AboveUnderPointCloudLoader
from dataset.augmentation import Normalize
from dataset.coordinate_utils import CylindricalCoordinates
from eval.utils import get_query_database_splits
from eval.vis_utils import submap_distance, print_token_similarity, remove_rt_attn_padding, get_rt_boundary_idx

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIG_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=BIG_SIZE)  # fontsize of the axes title

def load_eval_sets(params):
    eval_database_files, eval_query_files = get_query_database_splits(params)
    assert len(eval_database_files) == len(eval_query_files)

    # Just use default split for visualisation (Oxford - Oxford, AboveUnder - Karawatha)
    database_file = eval_database_files[0]
    query_file = eval_query_files[0]
    
    # Extract location name from query and database files
    if 'AboveUnder' in params.dataset_name or 'CSWildPlaces' in params.dataset_name:
        location_name = database_file.split('_')[1]
        temp = query_file.split('_')[1]
    else:
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
    assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                   query_file)
    p = os.path.join(params.dataset_folder, database_file)
    with open(p, 'rb') as f:
        database_sets = pickle.load(f)

    p = os.path.join(params.dataset_folder, query_file)
    with open(p, 'rb') as f:
        query_sets = pickle.load(f)
    return query_sets, database_sets

def process_pcl_to_octree(cloud: np.ndarray, params: TrainingParams) -> OctreeT:    
    cloud_tensor = torch.tensor(cloud, dtype=torch.float32)
    # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
    if params.normalize_points or params.scale_factor is not None:
        normalize_transform = Normalize(scale_factor=params.scale_factor,
                                        unit_sphere_norm=params.unit_sphere_norm)
        cloud_tensor = normalize_transform(cloud_tensor)
    if params.load_octree:  # Convert to Octree format
        # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
        mask = torch.all(abs(cloud_tensor) <= 1.0, dim=1)
        cloud_tensor = cloud_tensor[mask]
        # Also ensure this will hold if converting coordinate systems
        if params.model_params.coordinates == 'cylindrical':
            cloud_tensor_norm = torch.linalg.norm(cloud_tensor[:, :2], dim=1)[:, None]
            mask = torch.all(cloud_tensor_norm <= 1.0, dim=1)
            cloud_tensor = cloud_tensor[mask]
            # Convert to cylindrical coords
            coord_converter = CylindricalCoordinates(use_octree=True)
            cloud_tensor = coord_converter(cloud_tensor)
    # Convert to ocnn Points object, then create Octree
    cloud_ocnn = Points(cloud_tensor)
    octree = Octree(params.octree_depth, full_depth=2)
    octree.build_octree(cloud_ocnn)
    octree = ocnn.octree.merge_octrees([octree])  # this must be called for CT generation to be valid
    octree.construct_all_neigh()
    return octree

def plot_rt_attn_per_block_head(
    feats_and_attn_maps: List[Dict],
    octree: OctreeT,
    num_blocks_viz: int = 4,
    num_heads_viz: int = 6
):
    """
    Plot attention maps of multi-scale relay token attention for multiple blocks
    and heads.
    """
    num_hotf_blocks = len(feats_and_attn_maps)
    feats_and_attn_maps_block_i = feats_and_attn_maps[BLOCK_VIZ_IDX]
    pyramid_depths = octree.pyramid_depths
    B, H, N, _ = feats_and_attn_maps_block_i['rt_attn']['attn_map'].shape
    if not args.softmax:
        softmax_title_part = "maps (before softmax)"
    else:
        softmax_title_part = "maps (after softmax)"
    block_indices_viz = np.linspace(
        0, num_hotf_blocks-1, num_blocks_viz, dtype=np.int32,
    )
    head_indices_viz = np.linspace(
        0, H-1, num_heads_viz, dtype=np.int32
    )
    rt_boundary_idx = [
        (octree.batch_num_windows[depth_j].item()
         - torch.count_nonzero(octree.ct_batch_idx[depth_j]).item())
        for depth_j in pyramid_depths
    ]
    rt_boundary_idx = np.cumsum(rt_boundary_idx).tolist()
    # fig = plt.figure(figsize=(14, 9))
    ROW_LABEL_PAD = 5  # in pts
    fig, axes = plt.subplots(nrows=num_blocks_viz, ncols=num_heads_viz,
                             figsize=(10,9))
    fig.suptitle(
        # f"Multi-scale relay token attention {softmax_title_part} - per block and head, "
        f"RTSA attention {softmax_title_part}, "
        + f"{octree.num_pyramid_levels} pyramid levels",
        fontsize='x-large',
    )
    for i, block_idx in enumerate(block_indices_viz):
        if i >= num_blocks_viz:
            break
        rt_attn_map_i = feats_and_attn_maps[block_idx]['rt_attn']['attn_map'] # B, H, N, N
        if args.softmax:
            rt_attn_map_i = F.softmax(rt_attn_map_i, dim=-1)
        # Loop through heads
        for j, head_idx in enumerate(head_indices_viz):
            if j >= num_heads_viz:
                break
            ax = axes[i,j]
            # ax = fig.add_subplot(num_blocks_viz, num_heads_viz,
            #                      index=i*num_blocks_viz + j+1)
            rt_attn_map_head_i = rt_attn_map_i[0, head_idx]
            # Filter out mask tokens from attention map visualisation
            rt_attn_map_head_i = remove_rt_attn_padding(rt_attn_map_head_i, octree)

            #### rt_attn_map_head_i.masked_select(octree.rt_attn_mask[0] != octree.invalid_mask_value)
            #### Exclude padding attention values (which will be near zero)
            #### TODO: ALSO EXCLUDE MAX VALUES, OR JUST REMOVE PADDING FROM MAPS
            #### eps = 1e-6
            #### vmin = rt_attn_map[rt_attn_map > eps].min()
            #### vmin = np.median(rt_attn_map_head_i)
            
            # MATPLOTLIB VERSION:
            # temp = axes[i,j].imshow(rt_attn_map_head_i, vmin=vmin)
            temp = ax.imshow(rt_attn_map_head_i)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(temp, cax=cax)
            # fig.colorbar(temp, ax=ax, label=None)  # ENABLE COLORBAR
            if i == 0:
                ax.set_title(f"Head {head_idx+1}")
            if i == len(block_indices_viz) - 1:
                ax.set_xlabel('Level')
            # ax.set_xticks([], [])
            # ax.set_yticks([], [])
            # ax.set_xticks([0,*rt_boundary_idx[:2]], ['l=1','l=2','l=3'])
            # ax.set_yticks([0,*rt_boundary_idx[:2]], ['l=1','l=2','l=3'])
            ax.set_xticks([0,*rt_boundary_idx[:2]], [1, 2, 3])
            ax.set_yticks([0,*rt_boundary_idx[:2]], [1, 2, 3])
            if j == 0:
                ax.set_ylabel('Level', rotation=90)
                ax.annotate(
                    f"Block {block_idx+1}", xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - ROW_LABEL_PAD, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=BIG_SIZE, ha='right', va='center',
            )
        fig.tight_layout()
        
def plot_hosa_attn_per_block_head(
    feats_and_attn_maps: List[Dict],
    octree: OctreeT,
    num_blocks_viz: int = 4,
    num_heads_viz: int = 6
):
    """
    Plot attention maps of H-OSA for multiple blocks and heads.
    TODO: Visualise multiple depths
    """
    LEVEL = 1  # specify the pyramid level to extract visualise local attention from
    WINDOW_IDX = 0  # specify the attn window to visualise
    num_hotf_blocks = len(feats_and_attn_maps)
    feats_and_attn_maps_block_i = feats_and_attn_maps[BLOCK_VIZ_IDX]
    pyramid_depths = octree.pyramid_depths
    
    depth = pyramid_depths[LEVEL-1]
    B, H, N, _ = feats_and_attn_maps_block_i['local_attn'][depth]['attn_map'].shape
    if not args.softmax:
        softmax_title_part = "maps (before softmax)"
    else:
        softmax_title_part = "maps (after softmax)"
    block_indices_viz = np.linspace(
        0, num_hotf_blocks-1, num_blocks_viz, dtype=np.int32,
    )
    head_indices_viz = np.linspace(
        0, H-1, num_heads_viz, dtype=np.int32
    )
    # fig = plt.figure(figsize=(14, 9))
    ROW_LABEL_PAD = 5  # in pts
    fig, axes = plt.subplots(nrows=num_blocks_viz, ncols=num_heads_viz,
                             figsize=(10,9))
    fig.suptitle(
        # f"H-OSA attention {softmax_title_part} - per block and head - level {LEVEL}",
        f"H-OSA attention {softmax_title_part} - level {LEVEL}",
        fontsize='x-large',
    )
    for i, block_idx in enumerate(block_indices_viz):
        if i >= num_blocks_viz:
            break
        hosa_attn_map_i = feats_and_attn_maps[block_idx]['local_attn'][depth]['attn_map'] # B, H, N, N
        if args.softmax:
            hosa_attn_map_i = F.softmax(hosa_attn_map_i, dim=-1)
        # Loop through heads
        for j, head_idx in enumerate(head_indices_viz):
            if j >= num_heads_viz:
                break
            ax = axes[i,j]
            # ax = fig.add_subplot(num_blocks_viz, num_heads_viz,
            #                      index=i*num_blocks_viz + j+1)
            if args.hosa_viz_mode == 'heads':
                hosa_attn_map_head_i = hosa_attn_map_i[WINDOW_IDX, head_idx]
                title_str = 'Head'
            elif args.hosa_viz_mode == 'windows':
                hosa_attn_map_head_i = hosa_attn_map_i[head_idx].mean(0) # note that the first dim is not head idx, but I was too lazy to change the variable names to accomodate these two modes
                title_str = 'Window'
            # # Filter out mask tokens from attention map visualisation
            # hosa_attn_map_head_i = remove_rt_attn_padding(hosa_attn_map_head_i, octree)

            # MATPLOTLIB VERSION:
            # temp = axes[i,j].imshow(rt_attn_map_head_i, vmin=vmin)
            temp = ax.imshow(hosa_attn_map_head_i)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(temp, cax=cax)
            # fig.colorbar(temp, ax=ax, label=None)  # ENABLE COLORBAR
            if i == 0:
                ax.set_title(f"{title_str} {head_idx+1}")
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            # ax.set_xlabel('k')
            # ax.set_ylabel('q', rotation=0)
            if j == 0:
                ax.annotate(
                    f"Block {block_idx+1}", xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - ROW_LABEL_PAD, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=BIG_SIZE, ha='right', va='center',
            )
        fig.tight_layout()

    # Viz RPE to confirm if attention is solely reliant on RPE
    fig, axes = plt.subplots(nrows=num_blocks_viz, ncols=num_heads_viz,
                             figsize=(10,9))
    fig.suptitle(
        # f"H-OSA attention {softmax_title_part} - per block and head - level {LEVEL}",
        f"H-OSA RPE - level {LEVEL}",
        fontsize='x-large',
    )
    for i, block_idx in enumerate(block_indices_viz):
        if i >= num_blocks_viz:
            break
        hosa_rpe_i = feats_and_attn_maps[block_idx]['local_attn'][depth]['rpe'] # B, H, N, N
        # Loop through heads
        for j, head_idx in enumerate(head_indices_viz):
            if j >= num_heads_viz:
                break
            ax = axes[i,j]
            # ax = fig.add_subplot(num_blocks_viz, num_heads_viz,
            #                      index=i*num_blocks_viz + j+1)
            if args.hosa_viz_mode == 'heads':
                hosa_rpe_head_i = hosa_rpe_i[WINDOW_IDX, head_idx]
                title_str = 'Head'
            elif args.hosa_viz_mode == 'windows':
                hosa_rpe_head_i = hosa_rpe_i[head_idx].mean(0) # note that the first dim is not head idx, but I was too lazy to change the variable names to accomodate these two modes
                title_str = 'Window'
            # # Filter out mask tokens from attention map visualisation
            # hosa_rpe_head_i = remove_rt_attn_padding(hosa_rpe_head_i, octree)

            # MATPLOTLIB VERSION:
            # temp = axes[i,j].imshow(rt_attn_map_head_i, vmin=vmin)
            temp = ax.imshow(hosa_rpe_head_i)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(temp, cax=cax)
            # fig.colorbar(temp, ax=ax, label=None)  # ENABLE COLORBAR
            if i == 0:
                ax.set_title(f"{title_str} {head_idx+1}")
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            # ax.set_xlabel('k')
            # ax.set_ylabel('q', rotation=0)
            if j == 0:
                ax.annotate(
                    f"Block {block_idx+1}", xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - ROW_LABEL_PAD, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=BIG_SIZE, ha='right', va='center',
            )
        fig.tight_layout()

def plot_rt_attn_in_octree(
    rt_attn_map_i: Tensor,
    octree: OctreeT,
    pyramid_depths: List[int],
    block_idx: int,
    rt_viz_topk_idx: int,
    num_hotf_blocks: int,
):
    """
    Visualises the attention of multi-scale relay tokens within the octree.
    """
    # Get points from finest octree resolution in pyramid, and colourise by
    #   attn window index
    points_octree_per_depth = []
    windows_idx_per_depth = []
    for depth_j in pyramid_depths:
        points_octree, points_octree_windows, windows_idx = (
            get_octree_points_and_windows(octree, depth_j, params.model_params.quantizer)
        )
        points_octree_per_depth.append(points_octree)
        windows_idx_per_depth.append(windows_idx)

    # Get relay token boundaries and centroids
    rt_boundary_idx, rt_boundary_idx_cumsum = get_rt_boundary_idx(
        octree, pyramid_depths, return_cumsum=True
    )
    rt_centroids = [
        octree.window_stats[depth_j][:rt_boundary_idx[j],:3]
        for j, depth_j in enumerate(pyramid_depths)
    ]
    
    # Correct idx for the pyramid depth
    for j, rt_boundary in enumerate(rt_boundary_idx_cumsum):
        if rt_viz_topk_idx < rt_boundary:
            rt_topk_depth = pyramid_depths[j]
            rt_topk_idx_depth_j = rt_viz_topk_idx - (rt_boundary_idx_cumsum[j-1] if j > 0 else 0)
            break
    
    # Convert to list of hex colours
    # TODO: Add small change to colour every iteration, to make distinct windows
    #       look different
    patches_idx_colours_per_depth = []
    for j in range(len(pyramid_depths)):
        if args.octree_window_cmap == 'tab20':
            windows_idx_cmap = windows_idx_per_depth[j] % 20  # cap values at 20 for compatibility with tab20 colormap
            patches_idx_colours_per_depth.append([cm.to_hex(plt.cm.tab20(val)) for val in windows_idx_cmap])
        elif args.octree_window_cmap == 'tab10':
            windows_idx_cmap = windows_idx_per_depth[j] % 10  # cap values at 20 for compatibility with tab20 colormap
            patches_idx_colours_per_depth.append([cm.to_hex(plt.cm.tab10(val)) for val in windows_idx_cmap])
        else:
            raise ValueError(f"Unknown cmap type: {args.octree_window_cmap}")
    
    # Get point cloud boundaries of each octree depth to keep plots consistent
    ## lims = {}
    ## for i, coord in enumerate(['x','y','z']):
    ##     lims[coord] = [points_octree_per_depth[0][:,i].min(),
    ##                    points_octree_per_depth[0][:,i].max()]
    points_min_per_depth = []
    points_max_per_depth = []
    for j, points_octree_depth_j in enumerate(points_octree_per_depth):
        points_min_per_depth.append(points_octree_depth_j.min(0).values)
        points_max_per_depth.append(points_octree_depth_j.max(0).values)
    lims = torch.stack(
        [torch.stack(points_min_per_depth).min(0).values,
         torch.stack(points_max_per_depth).max(0).values,]
    ).T
    ticks = np.arange(-1, 1, 0.2)
    
    # Plot point cloud windows
    if args.plot_octree_windows:
        fig = plt.figure(figsize=(12, 4))
        fig.suptitle(f"Octree Attention Windows", fontsize='x-large')
        for j, depth_j in enumerate(pyramid_depths):
            ax = fig.add_subplot(1, 3, j+1, projection='3d')
            ax.view_init(elev=PLOT_ELEV, azim=PLOT_AZIM)
            _ = ax.scatter(*points_octree_per_depth[j].T, c=patches_idx_colours_per_depth[j],
                        alpha=0.7)
            # _ = ax.scatter(*points_octree_per_depth[0].T.numpy(), c='grey', alpha=0.2)
            ax.set_title(f'Pyramid level {j+1} (depth {depth_j})')
            ax.set_xticks(ticks, [])
            ax.set_yticks(ticks, [])
            ax.set_zticks(ticks, [])
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_zlim(*lims[2])
            ax.set_aspect('equal', adjustable='box')

    # Plot relay token attn scores within the octree
    fig = plt.figure(figsize=(12, 4))
    if args.average_attn_heads:
        heads_title_part = "heads averaged"
    else:
        heads_title_part = "single head"
    fig.suptitle(
        f"Multi-scale relay token attention maps - {heads_title_part} - "
        +f"block {list(range(num_hotf_blocks))[block_idx]+1}",
        fontsize='x-large'
    )
    vmin = rt_attn_map_i[:, rt_viz_topk_idx].min()
    vmax = rt_attn_map_i[:, rt_viz_topk_idx].max()
    axs = []
    for j, depth_j in enumerate(pyramid_depths):
        # ax = fig.add_subplot(2, 2, 1+j+1, projection='3d')
        ax = fig.add_subplot(1, 3, j+1, projection='3d')
        axs.append(ax)
        ax.view_init(elev=PLOT_ELEV, azim=PLOT_AZIM)
        lower_bnd = 0 if j == 0 else rt_boundary_idx_cumsum[j-1]
        upper_bnd = rt_boundary_idx_cumsum[j]
        rt_attn_depth_j_topk = rt_attn_map_i[lower_bnd:upper_bnd, rt_viz_topk_idx]
        points_octree_depth_j = points_octree_per_depth[j]
        windows_idx_depth_j = windows_idx_per_depth[j]
        # Exclude the topk token from the regular point visualisation
        if depth_j == rt_topk_depth:
            windows_idx_depth_j = windows_idx_depth_j[
                windows_idx_depth_j != rt_topk_idx_depth_j
            ]
            points_octree_depth_j = torch.cat(
                [points_octree_depth_j[:(rt_topk_idx_depth_j * octree.patch_size)],
                 points_octree_depth_j[((rt_topk_idx_depth_j+1) * octree.patch_size):]]
            )
        rt_attn_depth_j_topk_per_point = rt_attn_depth_j_topk[windows_idx_depth_j]
        # rt_attn_depth_j_topk_per_point = rt_attn_depth_j_topk[windows_idx_per_depth[j]]  # no exclusion
        temp = ax.scatter(
            *points_octree_depth_j.T, c=rt_attn_depth_j_topk_per_point,
            cmap=CMAP, s=30.0, alpha=0.7, vmin=vmin, vmax=vmax
        )
        if depth_j == rt_topk_depth:
            points_rt_topk_only = points_octree_per_depth[j][
                rt_topk_idx_depth_j * octree.patch_size
                : (rt_topk_idx_depth_j+1) * octree.patch_size
            ]
            # WARNING: BELOW DOESN'T WORK IF rt_attn_depth_j_topk_per_point EXCLUDES TOPK POINT
            # rt_attn_topk_only = rt_attn_depth_j_topk_per_point[  # plot topk token colourised by attn score
            #     rt_topk_idx_depth_j * octree.patch_size
            #     : (rt_topk_idx_depth_j+1) * octree.patch_size
            # ]
            _ = ax.scatter(
                *points_rt_topk_only.T,
                # c=[rt_attn_depth_j_topk[rt_topk_idx_depth_j]]*octree.patch_size,
                # c=rt_attn_topk_only,
                # vmin=vmin, vmax=vmax
                # edgecolors='r',
                c='r',
                marker='D',
                linewidths=1.2,
                # s=80.0,
                alpha=0.7,
            )
        ax.set_title(f'Pyramid level {j+1} (depth {depth_j})')
        ax.set_xticks(ticks, [])
        ax.set_yticks(ticks, [])
        ax.set_zticks(ticks, [])
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        ax.set_aspect('equal', adjustable='box')
        # if j == len(pyramid_depths) - 1:  # add colorbar to last fig
        #     cbar = fig.colorbar(temp, ax=axs, orientation='horizontal',
        #                         fraction=.1,
        #                         label='Attention Weight')
        #     cbar.set_ticks([])
    plt.tight_layout()
    # # Apparently this works when titght_layout doesnt:
    # plt.savefig('fig.png',bbox_inches='tight')

    ### OLD VERSION, PLOTTING RELAY TOKENS AS A SINGLE TRIANGLE EACH    
    # # Plot relay token attn
    # # vmin = rt_attn_map_i_heads_avg[rt_topk_idx, :].min()
    # # vmax = rt_attn_map_i_heads_avg[rt_topk_idx, :].max()
    # vmin = rt_attn_map_i[:, rt_viz_topk_idx].min()
    # vmax = rt_attn_map_i[:, rt_viz_topk_idx].max()
    # for j, depth_j in enumerate(pyramid_depths):
    #     ax = fig.add_subplot(2, 2, 1+j+1, projection='3d')
    #     lower_bnd = 0 if j == 0 else rt_boundary_idx_cumsum[j-1]
    #     upper_bnd = rt_boundary_idx_cumsum[j]
    #     # rt_attn_depth_j_topk = rt_attn_map_i_heads_avg[rt_topk_idx, lower_bnd:upper_bnd]
    #     rt_attn_depth_j_topk = rt_attn_map_i[lower_bnd:upper_bnd, rt_viz_topk_idx]
    #     temp = ax.scatter(*rt_centroids[j].T, c=rt_attn_depth_j_topk,
    #                     marker='^', s=50.0, alpha=0.8,
    #                     vmin=vmin, vmax=vmax)
    #     if rt_viz_topk_idx < upper_bnd and rt_viz_topk_idx >= lower_bnd:
    #         _ = ax.scatter(*rt_centroids[j][rt_topk_idx_depth_j],
    #                     c=rt_attn_depth_j_topk[rt_topk_idx_depth_j],
    #                     marker='D', edgecolors='r', linewidths=2.2,
    #                     s=120.0, alpha=1.0, vmin=vmin, vmax=vmax)
    #     ax.set_title(f'Relay Token Attention - Pyramid level {j+1}')
    #     ax.set_xlim(*lims['x'])
    #     ax.set_ylim(*lims['y'])
    #     ax.set_zlim(*lims['z'])
    #     ax.set_aspect('equal', adjustable='box')
    #     # fig.colorbar(temp, ax=ax, label='Attention score')
    # plt.tight_layout()

    
def process_submap(submap, model, device, params: TrainingParams):
    """
    Generate attention map visualisations from submap.
    """
    model_params = params.model_params
    # Get correct point cloud loader
    if params.dataset_name in ['Oxford','CSCampus3D']:
        pc_loader = PNVPointCloudLoader()
    elif 'AboveUnder' in params.dataset_name or 'WildPlaces' in params.dataset_name:
        pc_loader = AboveUnderPointCloudLoader()    
    pcl = pc_loader.read_pc(os.path.join(params.dataset_folder, submap['query']))
    
    # Pre-process pcl and convert to octree
    octree = process_pcl_to_octree(pcl, params)
    batch = {'octree': octree.to(device)}
    with torch.no_grad():        
        # Pass through model to get attn map
        out = model(batch, global_only=True)
        feats_and_attn_maps = out['feats_and_attn_maps']
    num_hotf_blocks = len(feats_and_attn_maps)
    feats_and_attn_maps_block_i = feats_and_attn_maps[BLOCK_VIZ_IDX]
    B, H, N, _ = feats_and_attn_maps_block_i['rt_attn']['attn_map'].shape

    # FEATS AND ATTN MAPS STRUCTURE:
    # feats_and_attn_maps : [{
    #     'local_attn': {depth_j: {attn_dict} for depth_j in pyramid_depths},
    #     'rt_attn': {'attn_map: ..., 'q': ..., 'k': ..., 'v':, ...},
    #     'local_feats': local_feat_dict,
    #     'rt_feats_pre_local': relay_token_dict,
    #     'rt_feats_post_local': relay_token_dict,
    # } for i in num_blocks]
    
    # TODO: Need to visualise two things. 
    #   A: The attention scores of relay tokens relative to their local
    #      windows. Can plot the whole point cloud with grey points, then plot
    #      the local window using colormap from attn map
    #   B: The attention scores of relay token global attention, for different
    #      relay tokens. Highlight the current window, and have colourmap for
    #      attn scores of other windows

    # feats_and_attn_maps_i = feats_and_attn_maps[BLOCK_IDX]
    
    # Print similarity of relay tokens (row-wise cosine sim, box plots)
    relay_tokens_i = feats_and_attn_maps_block_i['rt_feats_pre_local']
    local_feats_i = feats_and_attn_maps_block_i['local_feats']
    if not args.debug:  # BOX PLOTS
        print_token_similarity(relay_tokens_i, token_type='Relay')
        print_token_similarity(local_feats_i, token_type='Local')
    
    # Build window octree
    window_max_depth = octree.depth - model_params.num_input_downsamples
    num_stages = model_params.num_octf_levels + model_params.num_pyramid_levels
    octree = OctreeT(
        octree=octree, patch_size=model_params.patch_size,
        dilation=model_params.dilation, nempty=True,
        max_depth=window_max_depth,
        start_depth=window_max_depth-num_stages+1,
        rt_size=model_params.ct_size,
        ADaPE_mode=model_params.ADaPE_mode,
        ADaPE_use_accurate_point_stats=model_params.ADaPE_use_accurate_point_stats,
        num_pyramid_levels=model_params.num_pyramid_levels,
        num_octf_levels=model_params.num_octf_levels)
    octree.build_t()
    pyramid_depths = octree.pyramid_depths
    
    # Compute final idx of relay tokens (minus padding) from each depth
    rt_boundary_idx, rt_boundary_idx_cumsum = get_rt_boundary_idx(
        octree, pyramid_depths, return_cumsum=True
    )
    print(f"\tRT Boundary idx: {rt_boundary_idx_cumsum}")

    # Plot relay token attention maps per block and head
    # num_blocks_viz, num_heads_viz = 4, 4
    #!!!!!!!!!!!!! NOTE: UNCOMMENT THIS ONCE FINISHED DEBUGGING FOLLOWING BLOCKS !!!!!!!!!!!!!#
    if not args.debug:
        plot_rt_attn_per_block_head(
            feats_and_attn_maps, octree, NUM_BLOCKS_VIZ, NUM_HEADS_VIZ,
        )
        plot_hosa_attn_per_block_head(
            feats_and_attn_maps, octree, NUM_BLOCKS_VIZ, NUM_HEADS_VIZ,
        )

    # TODO: Plot relay token attention maps within the point cloud

    # TODO: Plot finest resolution octree (for comparison), then get RT centroids
    #       for all three depths, and plot these coloured by the attn maps.
    #       May be best to pick the RTs with highest avg attention scores, and
    #       visualise attn relative to them.

    # Visualise the first and last transformer blocks within the octree
    block_idxs = [-1, (num_hotf_blocks-1)//2, 0] if args.compare_first_last_blocks else [BLOCK_VIZ_IDX]
    rt_viz_topk_indices = []
    for block_idx in block_idxs:
        rt_attn_map_i = feats_and_attn_maps[block_idx]['rt_attn']['attn_map'][0]  # H, N, N
        if args.softmax:
            rt_attn_map_i = F.softmax(rt_attn_map_i, dim=-1)

        # Average over attention heads and filter out mask tokens
        if args.average_attn_heads:
            rt_attn_map_i = rt_attn_map_i.mean(0)
        else:
            rt_attn_map_i = rt_attn_map_i[HEAD_VIZ_IDX]
        rt_attn_map_i = remove_rt_attn_padding(rt_attn_map_i, octree)
        
        ### DEBUGGING: Plot avg attn map ###
        # if args.debug:
        #     fig = plt.figure(figsize=(4,4))
        #     ax = fig.add_subplot(1,1,1)
        #     ax.imshow(rt_attn_map_i)
        #     ax.set_xticks([0,*rt_boundary_idx_cumsum[:2]], ['d1','d2','d3'])
        #     ax.set_yticks([0,*rt_boundary_idx_cumsum[:2]], ['d1','d2','d3'])
        #     plt.tight_layout()
        ####################################

        # Check if relay token indices were already selected for the previous block
        if len(rt_viz_topk_indices) == 0:            
            # Get window idx by picking key tokens with highest avg top-k attn score
            topk_attn_vals = 5
            rt_attn_topk_scores = rt_attn_map_i.topk(topk_attn_vals, dim=0).values
            rt_attn_topk_indices = rt_attn_topk_scores.mean(dim=0).sort(descending=True).indices
            
            # Try pick highly attended-to tokens that are distant from each other
            rt_dist_threshold = 0.3  # (-1, 1) point range
            rt_prev_centroid, rt_prev_depth = None, None
            for rt_topk_idx in rt_attn_topk_indices:
                rt_depth = None
                rt_dist = torch.inf
                # Remove corresponding boundary idx cumsum to correct the idx for certain objects
                for j, rt_boundary in enumerate(rt_boundary_idx_cumsum):
                    if rt_topk_idx < rt_boundary:
                        rt_depth = pyramid_depths[j]
                        rt_topk_idx_depth_j = rt_topk_idx - (rt_boundary_idx_cumsum[j-1] if j > 0 else 0)
                        break
                rt_centroid = octree.window_stats[rt_depth][rt_topk_idx_depth_j,:3]
                if rt_prev_centroid is not None:
                    rt_dist = (rt_centroid - rt_prev_centroid).norm()
                # Filter by distance, or if from different level of pyramid
                if rt_dist > rt_dist_threshold or rt_depth != rt_prev_depth:
                    rt_viz_topk_indices.append(rt_topk_idx)
                    rt_prev_centroid = rt_centroid
                    rt_prev_depth = rt_depth
                    if len(rt_viz_topk_indices) >= NUM_RT_ATTN_VIZ:
                        break
        if len(rt_viz_topk_indices) < NUM_RT_ATTN_VIZ:
            print(f"[WARNING] only {len(rt_viz_topk_indices)}/{NUM_RT_ATTN_VIZ} "
                + f"relay tokens selected, reduce distance threshold!")
        print(f"\tVisualising top-k tokens: {rt_viz_topk_indices}")
        
        # Plot each set of relay tokens
        for i, rt_viz_topk_idx in enumerate(rt_viz_topk_indices):
            plot_rt_attn_in_octree(
                rt_attn_map_i=rt_attn_map_i,
                octree=octree,
                pyramid_depths=pyramid_depths,
                block_idx=block_idx,
                rt_viz_topk_idx=rt_viz_topk_idx,
                num_hotf_blocks=num_hotf_blocks,
            )

    return
    
    # for i, depth in enumerate(feats_and_attn_maps.keys()):
    #     points_octree, points_octree_windows, windows_idx = \
    #         get_octree_points_and_windows(octree, depth, params.model_params.quantizer)
    #     num_windows = len(points_octree_windows)
    #     num_blocks = len(feats_and_attn_maps[depth])

    #     # Get attn maps for depth
    #     local_attn_map = feats_and_attn_maps[depth][BLOCK_VIZ_IDX]['local_attn']['attn_map'] # N, H, CT+K, CT+K
    #     if args.softmax:
    #         local_attn_map = F.softmax(local_attn_map, dim=-1)
    #     num_heads = local_attn_map.size(1)
    #     # Avg over heads
    #     local_attn_map = torch.mean(local_attn_map, 1)
    #     ct_attn_map = None
    #     if 'ct_attn' in feats_and_attn_maps[depth][BLOCK_VIZ_IDX].keys():
    #         ct_attn_map = feats_and_attn_maps[depth][BLOCK_VIZ_IDX]['ct_attn']['attn_map'] # B, H, N, N
    #         if args.softmax:
    #             ct_attn_map = F.softmax(ct_attn_map, dim=-1)
    #         # Avg over heads
    #         ct_attn_map = torch.mean(ct_attn_map, 1)
    #     # Plot point cloud in grey, then plot attn windows
    #     fig = plt.figure(figsize=(18, 9))
    #     if not args.softmax:
    #         softmax_title_part = "maps (before softmax)"
    #     else:
    #         softmax_title_part = "maps (after softmax)"
    #     if ct_attn_map is not None:
    #         fig.suptitle(f"Window attention {softmax_title_part} - heads averaged, stage {i+1} (depth {depth})",
    #                      fontsize='x-large')
    #     else:
    #         fig.suptitle(f"Attention {softmax_title_part} of local windows - heads averaged, stage {i+1} (depth {depth})",
    #                      fontsize='x-large')
    #     for j, window_idx in enumerate(range(0, num_windows, max(num_windows//4, 1))):
    #         window_mask = windows_idx == window_idx
    #         points_octree_plot = points_octree[~window_mask]
    #         window_points = points_octree_windows[window_idx].T.numpy()
    #         window_attn_scores = local_attn_map[window_idx, 0, 1:].numpy() # idx 0 is CT, if using CTs
    #         window_attn_min = local_attn_map[window_idx].min()
    #         window_attn_max = local_attn_map[window_idx].max()
    #         ax = fig.add_subplot(2, 4, 2*j+1, projection='3d')
    #         _ = ax.scatter(*points_octree_plot.T.numpy(), c='grey', alpha=0.2)
    #         if ct_attn_map is not None:
    #             # Below plots how much the query CT attends to each local key for the current window
    #             temp = ax.scatter(*window_points, c=window_attn_scores, vmin=window_attn_min, vmax=window_attn_max, alpha=0.7)
    #             ax.set_title(f"Window attention to CT")
    #         else:  # local attn, highlight the first point and show attention for it
    #             temp = ax.scatter(*window_points[:, 1:], c=window_attn_scores, vmin=window_attn_min, vmax=window_attn_max, alpha=0.7)
    #             _ = ax.scatter(*window_points[:, 0], c='red', marker='D', alpha=1.0)
    #             ax.set_title(f"Window attention to first token (red)")
    #         # fig.colorbar(temp, ax=ax, label='Attention score')
    #         # ax.set_xlabel('x')
    #         # ax.set_ylabel('y')
    #         # ax.set_zlabel('z')
    #         ax.set_aspect('equal', adjustable='box')

    #         # Plot local attn map
    #         ax = fig.add_subplot(2, 4, 2*j+1+1)
    #         temp = ax.imshow(local_attn_map[window_idx].numpy())
    #         fig.colorbar(temp, ax=ax, label='Attention score')
    #         if ct_attn_map is not None:
    #             ax.set_title(f"Attention map of local window (CT is first elem)")
    #         else:
    #             ax.set_title(f"Attention map of local window")
    #         ax.axes.get_xaxis().set_ticks([])
    #         ax.axes.get_yaxis().set_ticks([])
    #         ax.set_xlabel('k')
    #         ax.set_ylabel('q', rotation=0)
    #     plt.tight_layout()
                
    #     if ct_attn_map is not None:
    #         # Plot ct global attn maps
    #         fig = plt.figure(figsize=(11, 9))
    #         fig.suptitle(f"CT Global Attention Map - depth {depth}")
    #         for j, head_idx in enumerate(range(0, num_heads, max(num_heads//4, 1))):
    #             ax = fig.add_subplot(2, 2, j+1)
    #             # temp = ax.imshow(ct_attn_map[0, head_idx].numpy(), vmin=-1, vmax=1)
    #             temp = ax.imshow(ct_attn_map[0, head_idx].numpy())
    #             fig.colorbar(temp, ax=ax, label='Attention score')
    #             ax.set_title(f"Head {head_idx}")
    #             ax.axes.get_xaxis().set_ticks([])
    #             ax.axes.get_yaxis().set_ticks([])
    #             ax.set_xlabel('k')
    #             ax.set_ylabel('q', rotation=0)

    #     ## VISUALISE ALL ATTN HEADS

    #     ## VISUALISE ATTN HEADS FOR 4 BLOCKS IN STAGE 2
    #     if i == 1 or len(feats_and_attn_maps.keys()) == 1:
    #         # TODO: Get attn maps for each block
    #         fig = plt.figure(figsize=(12, 10))
    #         fig.suptitle(f"Window attention maps {'after' if args.softmax else 'before'} softmax, stage {i+1} (depth {depth})",
    #                      fontsize="x-large")
    #         block_idx_list = [0, 6, 12, 16]
    #         for j, block_idx in enumerate(block_idx_list):
    #             block_local_attn_map = feats_and_attn_maps[depth][block_idx]['local_attn']['attn_map'] # N, H, CT+K, CT+K
    #             if args.softmax:
    #                 block_local_attn_map = F.softmax(block_local_attn_map, dim=-1)
    #             ### # Avg over heads
    #             ### block_local_attn_map = torch.mean(block_local_attn_map, 1)
    #             # Plot local attn map
    #             for k, head_idx in enumerate(range(0, num_heads, max(num_heads//4, 1))):
    #                 # Get desired head
    #                 block_head_local_attn_map = block_local_attn_map[:, head_idx]
    #                 ax = fig.add_subplot(4, 4, j*4+k+1)
    #                 temp = ax.imshow(block_head_local_attn_map[window_idx].numpy())
    #                 # fig.colorbar(temp, ax=ax, label='Attention score')
    #                 ax.set_title(f"Block {block_idx+1}")
    #                 ax.axes.get_xaxis().set_ticks([])
    #                 ax.axes.get_yaxis().set_ticks([])
    #                 ax.set_xlabel('k')
    #                 ax.set_ylabel('q', rotation=0)
    #         plt.tight_layout()

def main(model, device, params: TrainingParams, num_positives: int):
    global SET_IDX, QUERY_IDX, BLOCK_VIZ_IDX, HEAD_VIZ_IDX, NUM_RT_ATTN_VIZ
    global NUM_BLOCKS_VIZ, NUM_HEADS_VIZ, PLOT_ELEV, PLOT_AZIM, CMAP
    # SET_IDX = 0
    SET_IDX = 1
    QUERY_IDX = 0
    # QUERY_IDX = 10
    BLOCK_VIZ_IDX = -1
    NUM_RT_ATTN_VIZ = 1
    # local & rt attn viz
    NUM_BLOCKS_VIZ = 4
    NUM_HEADS_VIZ = 4
    HEAD_VIZ_IDX = 0
    PLOT_ELEV = 22
    PLOT_AZIM = 35
    # PLOT_AZIM = -35
    CMAP = 'viridis'
    
    model_params = params.model_params
    model.to(device)
    model.eval()
    
    # Get queries and database for desired split
    query_sets, database_sets = load_eval_sets(params)

    # Sample a query
    query = query_sets[SET_IDX][QUERY_IDX]

    # Sample a positive
    positive_sets = list(range(len(database_sets)))
    positive_sets.remove(SET_IDX)
    positives = []
    # FIXME: will loop infinitely if no positives remain
    while len(positives) < num_positives:
        database_idx = random.choice(positive_sets)
        if len(query[database_idx]) > 0:
            positive_idx = random.choice(query[database_idx])
            temp_positive = database_sets[database_idx][positive_idx]
            if temp_positive not in positives:
                # Print distance between query and positive
                pos_dist = submap_distance(query, temp_positive)
                print(f"Positive {len(positives)} distance: {pos_dist:.2f}m")
                positives.append(temp_positive)
                break

    # Process attention maps and other visualisations for query and positives
    print("Processing query:")
    process_submap(query, model, device, params)
    for i, positive in enumerate(positives):
        print(f"\nProcessing positive {i}:")
        process_submap(positive, model, device, params)

    plt.show()

    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise embeddings for positives and negatives.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False,
                        help='Trained model weights')
    parser.add_argument('--num_positives', type=int, default=1,
                        help='Number of positives to sample')
    parser.add_argument('--softmax', action='store_true',
                        help='Apply softmax to attn maps')
    parser.add_argument('--octree_window_cmap', type=str, default='tab10', choices=['tab10', 'tab20'],
                        help='Colourmap to use for visualising octree attn windows')
    parser.add_argument('--average_attn_heads', action='store_true',
                        help='Average attention scores over all heads (when visualising attn on the point cloud). If disabled, a single attn head is picked instead.')
    parser.add_argument('--hosa_viz_mode', type=str, default='heads', choices=['heads', 'windows'],
                        help='Whether to visualise a single hosa window over different heads, or multiple hosa windows with averaged heads')
    parser.add_argument('--compare_first_last_blocks', action='store_true',
                        help='Compare the attn scores within the octree from frst and last blocks')
    parser.add_argument('--plot_octree_windows', action='store_true',
                        help='Also visualise each set of octree windows alongside attention plots')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode. Temporarily disables plots I deem annoying while debugging.')

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('')

    set_seed()  # Seed RNG
    print('Determinism: Enabled')
    params = TrainingParams(args.config, args.model_config)
    # ensure attn maps are returned for this visualisation
    params.model_params.return_feats_and_attn_maps = True
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        if os.path.splitext(args.weights)[1] == '.ckpt':
            state = torch.load(args.weights)
            model.load_state_dict(state['model_state_dict'])
        else:  # .pt or .pth
            model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    main(model, device, params, args.num_positives)
    