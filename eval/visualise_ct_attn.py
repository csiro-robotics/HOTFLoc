"""
Visualise carrier token attention maps.

Ethan Griffiths (QUT & CSIRO Data61).
"""

import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import random
from tqdm import tqdm
import ocnn
from ocnn.octree import Octree, Points
from models.octree import OctreeT, rescale_octree_points
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as cm

from models.model_factory import model_factory
from misc.utils import TrainingParams
from misc.torch_utils import set_seed
from dataset.pointnetvlad.pnv_raw import PNVPointCloudLoader
from dataset.AboveUnder.AboveUnder_raw import AboveUnderPointCloudLoader
from dataset.augmentation import Normalize
from dataset.coordinate_utils import CylindricalCoordinates
from eval.utils import get_query_database_splits

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIG_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title

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


def main(model, device, params: TrainingParams):
    SET_IDX = 0
    QUERY_IDX = 0
    # SET_IDX = 2
    # QUERY_IDX = 1000
    BLOCK_IDX = -2  # last block will be dilated, so take second to last
    
    model_params = params.model_params
    model.to(device)
    model.eval()  # Eval mode

    # Get correct point cloud loader
    if params.dataset_name in ['Oxford','CSCampus3D']:
        pc_loader = PNVPointCloudLoader()
    elif 'AboveUnder' in params.dataset_name or 'WildPlaces' in params.dataset_name:
        pc_loader = AboveUnderPointCloudLoader()
    
    # Get queries and database for desired split
    query_sets, database_sets = load_eval_sets(params)

    # Sample a query (#TODO loop through queries, or get a positive too)
    query = query_sets[SET_IDX][QUERY_IDX]
    query_pc = pc_loader.read_pc(os.path.join(params.dataset_folder, query['query']))

    # Pre-process pcl and convert to octree
    query_octree = process_pcl_to_octree(query_pc, params)
    batch = {'octree': query_octree.to(device)}
    with torch.no_grad():        
        # Pass through model to get attn map
        out = model(batch, global_only=True)
        feats_and_attn_maps = out['feats_and_attn_maps']

    # TODO: Need to visualise two things. 
    #   A: The attention scores of carrier tokens relative to their local
    #      windows. Can plot the whole point cloud with grey points, then plot
    #      the local window using colormap from attn map
    #   B: The attention scores of carrier token global attention, for different
    #      carrier tokens. Highlight the current window, and have colourmap for
    #      attn scores of other windows

    # Build window octree
    window_max_depth = query_octree.depth - model_params.num_input_downsamples
    query_octree = OctreeT(octree=query_octree, patch_size=model_params.patch_size,
                           dilation=model_params.dilation, nempty=True,
                           max_depth=window_max_depth,
                           start_depth=window_max_depth-len(model_params.channels)+1,
                           ct_layers=model_params.ct_layers, ct_size=model_params.ct_size,
                           ADaPE_mode=model_params.ADaPE_mode,
                           ADaPE_use_accurate_point_stats=model_params.ADaPE_use_accurate_point_stats)
    query_octree.build_t()
    
    for i, depth in enumerate(feats_and_attn_maps.keys()):
        points_octree, points_octree_windows, windows_idx = \
            get_octree_points_and_windows(query_octree, depth, params.model_params.quantizer)
        num_windows = len(points_octree_windows)
        num_blocks = len(feats_and_attn_maps[depth])

        # Get attn maps for depth
        local_attn_map = feats_and_attn_maps[depth][BLOCK_IDX]['local_attn']['attn_map'] # N, H, CT+K, CT+K
        if args.softmax:
            local_attn_map = torch.nn.functional.softmax(local_attn_map, dim=-1)
        num_heads = local_attn_map.size(1)
        # Avg over heads
        local_attn_map = torch.mean(local_attn_map, 1)
        ct_attn_map = None
        if 'ct_attn' in feats_and_attn_maps[depth][BLOCK_IDX].keys():
            ct_attn_map = feats_and_attn_maps[depth][BLOCK_IDX]['ct_attn']['attn_map'] # B, H, N, N
            if args.softmax:
                ct_attn_map = torch.nn.functional.softmax(ct_attn_map, dim=-1)
            # Avg over heads
            ct_attn_map = torch.mean(ct_attn_map, 1)
        # Plot point cloud in grey, then plot attn windows
        fig = plt.figure(figsize=(18, 9))
        if not args.softmax:
            softmax_title_part = "maps (before softmax)"
        else:
            softmax_title_part = "maps (after softmax)"
        if ct_attn_map is not None:
            fig.suptitle(f"Window attention {softmax_title_part} - heads averaged, stage {i+1} (depth {depth})",
                         fontsize='x-large')
        else:
            fig.suptitle(f"Attention {softmax_title_part} of local windows - heads averaged, stage {i+1} (depth {depth})",
                         fontsize='x-large')
        for j, window_idx in enumerate(range(0, num_windows, max(num_windows//4, 1))):
            window_mask = windows_idx == window_idx
            points_octree_plot = points_octree[~window_mask]
            window_points = points_octree_windows[window_idx].T.numpy()
            window_attn_scores = local_attn_map[window_idx, 0, 1:].numpy() # idx 0 is CT, if using CTs
            window_attn_min = local_attn_map[window_idx].min()
            window_attn_max = local_attn_map[window_idx].max()
            ax = fig.add_subplot(2, 4, 2*j+1, projection='3d')
            _ = ax.scatter(*points_octree_plot.T.numpy(), c='grey', alpha=0.2)
            if ct_attn_map is not None:
                # Below plots how much the query CT attends to each local key for the current window
                temp = ax.scatter(*window_points, c=window_attn_scores, vmin=window_attn_min, vmax=window_attn_max, alpha=0.7)
                ax.set_title(f"Window attention to CT")
            else:  # local attn, highlight the first point and show attention for it
                temp = ax.scatter(*window_points[:, 1:], c=window_attn_scores, vmin=window_attn_min, vmax=window_attn_max, alpha=0.7)
                _ = ax.scatter(*window_points[:, 0], c='red', marker='D', alpha=1.0)
                ax.set_title(f"Window attention to first token (red)")
            # fig.colorbar(temp, ax=ax, label='Attention score')
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z')
            ax.set_aspect('equal', adjustable='box')

            # Plot local attn map
            ax = fig.add_subplot(2, 4, 2*j+1+1)
            temp = ax.imshow(local_attn_map[window_idx].numpy())
            fig.colorbar(temp, ax=ax, label='Attention score')
            if ct_attn_map is not None:
                ax.set_title(f"Attention map of local window (CT is first elem)")
            else:
                ax.set_title(f"Attention map of local window")
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_xlabel('k')
            ax.set_ylabel('q', rotation=0)
        plt.tight_layout()

        # ##### ALSO PLOT RPE: ####
        # # Get rpe maps for depth
        # local_attn_rpe = feats_and_attn_maps[depth][BLOCK_IDX]['local_attn']['rpe'] # N, H, CT+K, CT+K
        # num_heads = local_attn_rpe.size(1)
        # # Avg over heads
        # local_attn_rpe = torch.mean(local_attn_rpe, 1)
        # # Plot point cloud in grey, then plot attn windows
        # fig = plt.figure(figsize=(18, 9))
        # fig.suptitle(f"RPE of local window - heads averaged, stage {i+1} (depth {depth})",
        #              fontsize='x-large')
        # for j, window_idx in enumerate(range(0, num_windows, max(num_windows//4, 1))):
        #     window_mask = windows_idx == window_idx
        #     points_octree_plot = points_octree[~window_mask]
        #     window_points = points_octree_windows[window_idx].T.numpy()
        #     window_rpe_scores = local_attn_rpe[window_idx, 0, 1:].numpy() # idx 0 is CT, if using CTs
        #     window_rpe_min = local_attn_rpe[window_idx].min()
        #     window_rpe_max = local_attn_rpe[window_idx].max()
        #     ax = fig.add_subplot(2, 4, 2*j+1, projection='3d')
        #     _ = ax.scatter(*points_octree_plot.T.numpy(), c='grey', alpha=0.2)
        #     # local rpe, highlight the first point and show attention for it
        #     temp = ax.scatter(*window_points[:, 1:], c=window_rpe_scores, vmin=window_rpe_min, vmax=window_rpe_max, alpha=0.7)
        #     _ = ax.scatter(*window_points[:, 0], c='red', marker='D', alpha=1.0)
        #     ax.set_title(f"RPE w.r.t first token (red)")
        #     ax.set_aspect('equal', adjustable='box')

        #     # Plot local attn map
        #     ax = fig.add_subplot(2, 4, 2*j+1+1)
        #     temp = ax.imshow(local_attn_rpe[window_idx].numpy())
        #     fig.colorbar(temp, ax=ax, label='RPE weighting')
        #     ax.set_title(f"RPE of local window")
        #     ax.axes.get_xaxis().set_ticks([])
        #     ax.axes.get_yaxis().set_ticks([])
        #     ax.set_xlabel('k')
        #     ax.set_ylabel('q', rotation=0)
        # plt.tight_layout()

        # #########################
                
        # if ct_attn_map is not None:
        #     # Plot ct global attn maps
        #     fig = plt.figure(figsize=(11, 9))
        #     fig.suptitle(f"CT Global Attention Map - depth {depth}")
        #     for j, head_idx in enumerate(range(0, num_heads, max(num_heads//4, 1))):
        #         ax = fig.add_subplot(2, 2, j+1)
        #         # temp = ax.imshow(ct_attn_map[0, head_idx].numpy(), vmin=-1, vmax=1)
        #         temp = ax.imshow(ct_attn_map[0, head_idx].numpy())
        #         fig.colorbar(temp, ax=ax, label='Attention score')
        #         ax.set_title(f"Head {head_idx}")
        #         ax.axes.get_xaxis().set_ticks([])
        #         ax.axes.get_yaxis().set_ticks([])
        #         ax.set_xlabel('k')
        #         ax.set_ylabel('q', rotation=0)

        ## VISUALISE ALL ATTN HEADS

        ## VISUALISE ATTN HEADS FOR 4 BLOCKS IN STAGE 2
        ROW_LABEL_PAD = 5  # in pts
        if i == 1 or len(feats_and_attn_maps.keys()) == 1:
            # TODO: Get attn maps for each block
            fig = plt.figure(figsize=(12, 10))
            fig.suptitle(f"Window attention maps {'after' if args.softmax else 'before'} softmax, stage {i+1} (depth {depth})",
                         fontsize="x-large")
            block_idx_list = [0, 6, 12, 16]
            for j, block_idx in enumerate(block_idx_list):
                block_local_attn_map = feats_and_attn_maps[depth][block_idx]['local_attn']['attn_map'] # N, H, CT+K, CT+K
                if args.softmax:
                    block_local_attn_map = torch.nn.functional.softmax(block_local_attn_map, dim=-1)
                ### # Avg over heads
                ### block_local_attn_map = torch.mean(block_local_attn_map, 1)
                # Plot local attn map
                for k, head_idx in enumerate(range(0, num_heads, max(num_heads//4, 1))):
                    # Get desired head
                    block_head_local_attn_map = block_local_attn_map[:, head_idx]
                    ax = fig.add_subplot(4, 4, j*4+k+1)
                    temp = ax.imshow(block_head_local_attn_map[window_idx].numpy())
                    # fig.colorbar(temp, ax=ax, label='Attention score')
                    if j == 0:
                        ax.set_title(f"Head {head_idx+1}")
                    # ax.set_title(f"Block {block_idx+1}")
                    ax.axes.get_xaxis().set_ticks([])
                    ax.axes.get_yaxis().set_ticks([])
                    # ax.set_xlabel('k')
                    # ax.set_ylabel('q', rotation=0)
                    if k == 0:
                        ax.annotate(
                            f"Block {int(np.ceil((block_idx+1)/18*10))}", xy=(0, 0.5),
                            xytext=(-ax.yaxis.labelpad - ROW_LABEL_PAD, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            ha='right', va='center',
                            # size='large', ha='right', va='center',
                    )
            plt.tight_layout()

            # Also plot RPE per head
            fig = plt.figure(figsize=(12, 10))
            fig.suptitle(f"Window attention RPE, stage {i+1} (depth {depth})",
                         fontsize="x-large")
            for j, block_idx in enumerate(block_idx_list):
                block_local_attn_rpe = feats_and_attn_maps[depth][block_idx]['local_attn']['rpe'] # N, H, CT+K, CT+K
                # Plot local attn rpe
                for k, head_idx in enumerate(range(0, num_heads, max(num_heads//4, 1))):
                    # Get desired head
                    block_head_local_attn_rpe = block_local_attn_rpe[:, head_idx]
                    ax = fig.add_subplot(4, 4, j*4+k+1)
                    temp = ax.imshow(block_head_local_attn_rpe[window_idx].numpy())
                    # fig.colorbar(temp, ax=ax, label='RPE Weighting')
                    if j == 0:
                        ax.set_title(f"Head {head_idx+1}")
                    # ax.set_title(f"Block {block_idx+1}")
                    ax.axes.get_xaxis().set_ticks([])
                    ax.axes.get_yaxis().set_ticks([])
                    # ax.set_xlabel('k')
                    # ax.set_ylabel('q', rotation=0)
                    if k == 0:
                        ax.annotate(
                            # f"Block {block_idx+1}", xy=(0, 0.5),
                            f"Block {int(np.ceil((block_idx+1)/18*10))}", xy=(0, 0.5),
                            xytext=(-ax.yaxis.labelpad - ROW_LABEL_PAD, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            ha='right', va='center',
                            # size='large', ha='right', va='center',
                    )
                    
            plt.tight_layout()

        # TODO: VISUALISE BEFORE AND AFTER RPE 
        
    plt.show()

    
    
    ## TODO: Visualise local and ct feature map similarity

    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise embeddings for positives and negatives.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--num_queries', type=int, default=20, help='Number of queries to sample')
    parser.add_argument('--softmax', action='store_true', help='Apply softmax to attn maps')

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
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    main(model, device, params)
    