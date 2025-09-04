"""
Count stats about octrees + relay tokens in HOTFormerLoc
"""
import argparse
import os

import tqdm
import numpy as np
import torch
from ocnn.octree import Octree, Points

from dataset.dataset_utils import make_dataloaders
from models.model_factory import model_factory
from models.octree import OctreeT
from misc.torch_utils import set_seed, release_cuda, to_device
from misc.utils import TrainingParams

SAVE_FILE = 'octree_stats.txt'

def setup_model():
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    # params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    model.print_info()

    model.to(device)
    model.eval()

    # Get val dataloader
    # params.val_batch_size = 256
    # params.num_workers = 0
    dataloaders = make_dataloaders(params, validation=True)
    dataloader = dataloaders['global_val']

    return params, device, model, dataloader

def main():
    params, device, model, dataloader = setup_model()

    # Don't need to run forward pass for this, so leave model off GPU
    model.cpu()

    num_points = []
    num_relay_tokens_dict = {}
    num_octants_dict = {}

    for ii, batch_dict in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if ii*params.val_batch_size >= args.max_samples > 0:
            break
        batch_dict = to_device(batch_dict, device, non_blocking=True)
        points: Points = batch_dict['batch']['points']
        octree: Octree = batch_dict['batch']['octree']
        num_points.extend(points.batch_npt.tolist())
        # Construct OctreeT
        if params.model_params.downsample_input_embeddings:
            depth = octree.depth - params.model_params.num_input_downsamples  # current octree depth
        num_stages = params.model_params.num_octf_levels + params.model_params.num_pyramid_levels
        octree = OctreeT(
            octree,
            params.model_params.patch_size,
            params.model_params.dilation,
            nempty=True,
            max_depth=depth,
            start_depth=depth - num_stages + 1,
            rt_size=params.model_params.ct_size,
            rt_class_token=params.model_params.rt_class_token,
            ADaPE_mode=params.model_params.ADaPE_mode,
            ADaPE_use_accurate_point_stats=params.model_params.ADaPE_use_accurate_point_stats,
            num_pyramid_levels=params.model_params.num_pyramid_levels,
            num_octf_levels=params.model_params.num_octf_levels,
        )
        octree.build_t(points=points)
        # Get number of non-padding RTs per batch (accounts for padded final batch elem)
        for depth in octree.pyramid_depths:
            rt_batch_idx_list = octree.ct_batch_idx[depth].split(octree.batch_num_windows[depth].tolist())
            num_rt_per_batch = [torch.count_nonzero(x == i).item() for i, x in enumerate(rt_batch_idx_list)]
            if depth not in num_relay_tokens_dict:
                num_relay_tokens_dict[depth] = []
            num_relay_tokens_dict[depth].extend(num_rt_per_batch)
        # Get number of non-empty octants (local feats) per depth
        all_octf_depths = range(octree.max_depth, octree.start_depth-1, -1)
        for depth in all_octf_depths:
            num_octants_per_batch = octree.batch_nnum_nempty[depth].tolist()
            if depth not in num_octants_dict:
                num_octants_dict[depth] = []
            num_octants_dict[depth].extend(num_octants_per_batch)

    samples_str = f'{args.max_samples} samples' if args.max_samples > 0 else 'all samples'
    log_str = f'STATS ({samples_str}):'
    log_str += f'\n  Config: {params.params_path}'
    log_str += f'\n  Model Config: {params.model_params_path}'
    log_str += '\n  (mean, std, min, max)'
    num_points = np.asarray(num_points)
    log_str += (f"\n  Pointcloud size: "
                f"{num_points.mean():.0f}, ±{num_points.std():.0f}, {num_points.min():.0f}, {num_points.max():.0f}")
    log_str += "\n  Relay Tokens:"
    for depth in num_relay_tokens_dict.keys():
        num_relay_tokens = np.asarray(num_relay_tokens_dict[depth])
        log_str += f"\n    Depth {depth}: "
        log_str += (f"{num_relay_tokens.mean():.1f}, ±{num_relay_tokens.std():.1f}, "
                    f"{num_relay_tokens.min():.1f}, {num_relay_tokens.max():.1f}")
    log_str += "\n  Octants (local feats):"
    for depth in num_octants_dict.keys():
        num_octants = np.asarray(num_octants_dict[depth])
        log_str += f"\n    Depth {depth}: "
        log_str += (f"{num_octants.mean():.1f}, ±{num_octants.std():.1f}, "
                    f"{num_octants.min():.1f}, {num_octants.max():.1f}")
    log_str += "\n\n"
    print(log_str)
    filename = os.path.join(args.save_dir, SAVE_FILE)
    with open(filename, 'a') as f:
        f.write(log_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise correspondences from HOTFormerMetricLoc')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--save_dir', type=str, default='.', help='Path to save stats to (defaults to current directory)')
    parser.add_argument('--max_samples', type=int, default=-1, help='Max number of samples from validation set to test (defaults to all).')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Save path: {}'.format(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)
    print('Max samples: {}'.format(args.max_samples))
    if not (args.max_samples == -1 or args.max_samples > 0):
        raise ValueError
    print('Debug mode: {}'.format(args.debug))
    print('')

    set_seed()  # Seed RNG
    main()