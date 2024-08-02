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
from models.octree import OctreeT
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as cm

from models.model_factory import model_factory
from misc.utils import TrainingParams, set_seed, rescale_octree_points
from dataset.pointnetvlad.pnv_raw import PNVPointCloudLoader
from dataset.AboveUnder.AboveUnder_raw import AboveUnderPointCloudLoader
from dataset.augmentation import Normalize
from eval.utils import get_query_database_splits

def process_pcl_to_octree(cloud: np.ndarray, params: TrainingParams) -> OctreeT:    
    cloud_tensor = torch.tensor(cloud, dtype=torch.float32)
    # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
    if params.normalize_points or params.scale_factor is not None:
        normalize_transform = Normalize(scale_factor=params.scale_factor,
                                        unit_sphere_norm=params.unit_sphere_norm)
        cloud_tensor = normalize_transform(cloud_tensor)    
    mask = torch.all(abs(cloud_tensor) <= 1.0, dim=1)
    cloud_tensor = cloud_tensor[mask]
    # Convert to ocnn Points object, then create Octree
    cloud_ocnn = Points(cloud_tensor)
    octree = Octree(params.octree_depth, full_depth=2)
    octree.build_octree(cloud_ocnn)
    # octree = OctreeT(  # this subclass provides patch functionality ontop of existing Octree
    #     octree=octree, patch_size=params.model_params.patch_size, dilation=params.model_params.dilation,
    #     nempty=True, max_depth=octree.depth, start_depth=octree.full_depth,
    #     )
    octree = ocnn.octree.merge_octrees([octree])  # this must be called for CT generation to be valid
    octree.construct_all_neigh()
    return octree

def main(model, device, params: TrainingParams):
    model_params = params.model_params
    model.to(device)
    model.eval()  # Eval mode

    # Get correct point cloud loader
    if params.dataset_name in ['Oxford','Campus3D']:
        pc_loader = PNVPointCloudLoader()
    elif 'AboveUnder' in params.dataset_name or 'WildPlaces' in params.dataset_name:
        pc_loader = AboveUnderPointCloudLoader()
    
    # Get queries and database for desired split
    eval_database_files, eval_query_files = get_query_database_splits(params)

    assert len(eval_database_files) == len(eval_query_files)

    # Just use default split for visualisation (Oxford - Oxford, AboveUnder - Karawatha)
    database_file = eval_database_files[0]
    query_file = eval_query_files[0]
    
    # Extract location name from query and database files
    if 'AboveUnder' in params.dataset_name:
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

    # Sample a query (#TODO loop through queries, or get a positive too)
    query = query_sets[0][0]
    query_pc = pc_loader.read_pc(os.path.join(params.dataset_folder, query['query']))

    # Pre-process pcl and convert to octree
    query_octree = process_pcl_to_octree(query_pc, params)
    batch = {'octree': query_octree.to(device)}
    with torch.no_grad():        
        # Pass through model to get attn map
        out = model(batch)
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
                            ADaPE_mode=model_params.ADaPE_mode)
    
    for depth in feats_and_attn_maps.keys():
        if 'ct' not in feats_and_attn_maps[depth]['attn'].keys():
            continue        
        ############### Get windows ################################
        query_octree.build_t()
        key = query_octree.key(depth, nempty=True)
        x, y, z, _ = ocnn.octree.key2xyz(key, depth)
        xyz = torch.stack([x, y, z], dim=1)
        # Convert octree point coords to original scale
        points_octree = rescale_octree_points(xyz, depth)
        # Create window partitions and get idx
        points_octree_windows = query_octree.data_to_windows(points_octree, depth, False)
        patches_idx = torch.zeros(points_octree_windows.shape[:-1],
                                dtype=torch.int32)
        num_windows = len(patches_idx)
        idx_values = torch.arange(num_windows, dtype=torch.int32).unsqueeze(-1)  # integer values corresponding to patch idx
        patches_idx += idx_values
        # Reverse patch operation and remove padding
        patches_idx = patches_idx.reshape(-1)
        patches_idx = query_octree.patch_reverse(patches_idx, depth)
        ##############################################################

        # Get attn maps for depth
        ct_attn_map = feats_and_attn_maps[depth]['attn']['ct'] # B, H, N, N
        local_attn_map = feats_and_attn_maps[depth]['attn']['local'] # N, H, CT+K, CT+K    
        # Plot point cloud in grey, then plot attn windows
        # window_idx = 0
        num_heads = local_attn_map.size(1)
        head_idx = 0
        # num_heads = local_attn_map.size(1)
        fig = plt.figure(figsize=(18, 9))
        fig.suptitle(f"Attention score of points to CTs per local window (head {head_idx}) - depth {depth}")
        # for i, head_idx in enumerate(range(0, num_heads, num_heads//4)):
        for j, window_idx in enumerate(range(0, num_windows, max(num_windows//4, 1))):
            window_mask = patches_idx == window_idx
            points_octree_plot = points_octree[~window_mask]
            window_points = points_octree_windows[window_idx].T.numpy()
            # Below gets how much each point attends to the CT for the current window
            window_ct_attn = local_attn_map[window_idx, head_idx, 1:, 0].numpy() # idx 0 is CT
            ax = fig.add_subplot(2, 4, 2*j+1, projection='3d')
            _ = ax.scatter(*points_octree_plot.T.numpy(), c='grey')
            temp = ax.scatter(*window_points, c=window_ct_attn, vmin=-1, vmax=1)
            # fig.colorbar(temp, ax=ax, label='Attention score')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f"Attention of window points to corresponding CT")

            ## TODO: PLOT ATTN MAPS
            ax = fig.add_subplot(2, 4, 2*j+1+1)
            temp = ax.imshow(local_attn_map[window_idx, head_idx].numpy(), vmin=-1, vmax=1)
            fig.colorbar(temp, ax=ax, label='Attention score')
            ax.set_title(f"Attention map for all local features (CT is first elem)")
            
        # Plot ct global attn maps
        fig = plt.figure(figsize=(11, 9))
        fig.suptitle(f"CT Global Attention Map - depth {depth}")
        for j, head_idx in enumerate(range(0, num_heads, max(num_heads//4, 1))):
            ax = fig.add_subplot(2, 2, j+1)
            temp = ax.imshow(ct_attn_map[0,head_idx].numpy(), vmin=-1, vmax=1)
            fig.colorbar(temp, ax=ax, label='Attention score')
            ax.set_title(f"Head {head_idx}")
        
        plt.tight_layout()
        
    plt.show()

    
    
    ## TODO: Visualise local and ct feature map similarity

    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise embeddings for positives and negatives.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--num_queries', type=int, default=20, help='Number of queries to sample')

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

    model = model_factory(params.model_params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    main(model, device, params)
    