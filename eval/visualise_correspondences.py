"""
Visualises correspondences from HOTFormerMetricLoc (only on validation set).
"""
import argparse
import logging
import os

import numpy as np
import torch

from dataset.dataset_utils import make_dataloaders
from eval.vis_utils import (
    colourise_points_by_similarity,
    visualise_coarse_correspondences,
    visualise_fine_correspondences,
    visualise_registration,
    visualise_LGR_initial_registration,
    visualise_similarity,
)
from models.model_factory import model_factory
from models.losses.geotransformer_loss import Evaluator
from misc.torch_utils import set_seed, release_cuda, to_device
from misc.utils import TrainingParams
from misc.logger import create_logger
from geotransformer.utils.visualization import (
    draw_point_to_node,
    draw_node_correspondences,
    get_colors_with_tsne
)

def setup_model():
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
    print('Loading weights: {}'.format(args.weights))
    if os.path.splitext(args.weights)[1] == '.ckpt':
        state = torch.load(args.weights)
        model.load_state_dict(state['model_state_dict'])
    else:  # .pt or .pth
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)
    model.eval()

    logging_level = 'DEBUG' if params.debug else 'INFO'
    create_logger(log_file=None, logging_level=logging_level)

    # Get val dataloader
    params.val_batch_size = 1
    params.local.batch_size = 1
    dataloaders = make_dataloaders(params, local=True, validation=True)
    dataloader = dataloaders['local_val']
    dataloader.shuffle = True  # NOTE: CANT SHUFFLE VAL LOADER, ONLY EVAL

    return params, device, model, dataloader

def main():
    params, device, model, dataloader = setup_model()

    # Get evaluator
    metric_loc_evaluator = Evaluator(params)

    for ii, local_batch in enumerate(dataloader):
        log_str = f"ID {ii}: "
        local_batch = to_device(local_batch, device, non_blocking=True, construct_octree_neigh=True)
        with torch.inference_mode():
            output_dict = model(local_batch)[0]

        batch_temp = {'transform': local_batch['transform'][0]}  # temp fix since loss func expects a single batch item
        eval_metrics = release_cuda(metric_loc_evaluator(output_dict, batch_temp))

        # Skip successes
        if args.failures and eval_metrics['RR'] > 0:
            continue

        anc_points_coarse = release_cuda(output_dict['anc_points_coarse'])
        anc_points_fine = release_cuda(output_dict['anc_points_fine'])
        anc_point_to_node = release_cuda(output_dict['anc_point_to_node'])
        anc_feats_coarse = release_cuda(output_dict['anc_feats_coarse'], to_numpy=True)
        anc_feats_coarse_pre_refinement = release_cuda(output_dict['anc_feats_coarse_pre_refinement'], to_numpy=True)
        anc_feats_fine = release_cuda(output_dict['anc_feats_fine'], to_numpy=True)
        pos_points_coarse = release_cuda(output_dict['pos_points_coarse'], to_numpy=True)
        pos_points_fine = release_cuda(output_dict['pos_points_fine'])
        pos_point_to_node = release_cuda(output_dict['pos_point_to_node'])
        pos_feats_coarse = release_cuda(output_dict['pos_feats_coarse'])
        pos_feats_coarse_pre_refinement = release_cuda(output_dict['pos_feats_coarse_pre_refinement'], to_numpy=True)
        pos_feats_fine = release_cuda(output_dict['pos_feats_fine'])
        gt_node_corr_indices = release_cuda(output_dict['gt_node_corr_indices'])
        gt_node_corr_overlaps = release_cuda(output_dict['gt_node_corr_overlaps'])  # NOTE: Only averaged on non-zero overlap patches
        anc_node_corr_indices = release_cuda(output_dict['anc_node_corr_indices'])
        pos_node_corr_indices = release_cuda(output_dict['pos_node_corr_indices'])
        node_corr_indices = torch.stack((anc_node_corr_indices, pos_node_corr_indices), dim=1)
        anc_node_corr_knn_masks = release_cuda(output_dict['anc_node_corr_knn_masks'])
        pos_node_corr_knn_masks = release_cuda(output_dict['pos_node_corr_knn_masks'])
        anc_corr_points = release_cuda(output_dict['anc_corr_points'])
        pos_corr_points = release_cuda(output_dict['pos_corr_points'])
        best_anc_corr_points = release_cuda(output_dict['best_anc_corr_points'])
        best_anc_corr_points = best_anc_corr_points[torch.nonzero(best_anc_corr_points)[:,0].unique()]
        best_pos_corr_points = release_cuda(output_dict['best_pos_corr_points'])
        best_pos_corr_points = best_pos_corr_points[torch.nonzero(best_pos_corr_points)[:,0].unique()]
        T_best_corr = release_cuda(output_dict['best_corr_transform'])
        matching_scores = release_cuda(output_dict['matching_scores'], to_numpy=True)
        corr_scores = release_cuda(output_dict['corr_scores'], to_numpy=True)
        num_corr_points = np.array(output_dict['num_corr_points'])
        T_gt = release_cuda(local_batch['transform'][0])
        T_estimated = release_cuda(output_dict['estimated_transform'])

        num_points_per_patch = torch.concat(
            (anc_node_corr_knn_masks.float().sum(1),
             pos_node_corr_knn_masks.float().sum(1),),
            dim=0,
        )
        pos_dist = T_gt[:3,3].norm().item()
        
        # Print metrics as well

        log_str += f'pos dist: {pos_dist:.2f}m'
        log_str += f' -- RTE: {eval_metrics['RTE']:.2f}m -- RRE: {eval_metrics['RRE']:.2f}deg'
        log_str += f' -- Coarse IR: {eval_metrics['PIR']*100:.1f}% -- Fine IR: {eval_metrics['IR']*100:.1f}%'
        log_str += f' -- mean pts per patch: {num_points_per_patch.mean():.1f} (±{num_points_per_patch.std():.1f})'
        log_str += f' -- mean patch overlap: {gt_node_corr_overlaps.mean()*100:.1f}% (±{gt_node_corr_overlaps.std()*100:.1f}%)'
        log_str += f' -- num corr patches (after OT): {len(num_corr_points)}'
        log_str += f' -- mean corr pts per patch: {num_corr_points.mean():.1f} (±{num_corr_points.std():.1f})'
        log_str += f' -- mean corr score: {corr_scores.mean():.3f} (±{corr_scores.std():.3f})'

        print(log_str, flush=True)

        save_dir_ii = None
        if args.save_dir is not None:
            save_dir_ii = os.path.join(args.save_dir, f'{ii}')
            os.makedirs(save_dir_ii, exist_ok=True)

        visualise_coarse_correspondences(
            anc_points_coarse=anc_points_coarse,
            pos_points_coarse=pos_points_coarse,
            anc_points_fine=anc_points_fine,
            pos_points_fine=pos_points_fine,
            node_corr_indices=node_corr_indices.numpy(),
            gt_node_corr_indices=gt_node_corr_indices.numpy(),
            transform=T_gt.numpy(),
            anc_point_to_node=anc_point_to_node,
            pos_point_to_node=pos_point_to_node,
            anc_feats_coarse=anc_feats_coarse,
            pos_feats_coarse=pos_feats_coarse,
            translate=[0,0,50],
            # coarse_colourmode='patch',
            # coarse_colourmode='tsne',
            coarse_colourmode='umap',
            save_dir=save_dir_ii,
            disable_animation=args.disable_animation,
            non_interactive=args.non_interactive,
        )

        visualise_similarity(
            anc_points_fine=anc_points_fine,
            pos_points_fine=pos_points_fine,
            transform=T_gt.numpy(),
            anc_point_to_node=anc_point_to_node,
            pos_point_to_node=pos_point_to_node,
            anc_feats_coarse=anc_feats_coarse_pre_refinement,
            pos_feats_coarse=pos_feats_coarse_pre_refinement,
            translate=[0,0,50],
            # coarse_colourmode='tsne',
            coarse_colourmode='umap',
            save_dir=save_dir_ii,
            disable_animation=args.disable_animation,
            non_interactive=args.non_interactive,
        )

        visualise_fine_correspondences(
            anc_points_fine=anc_points_fine,
            pos_points_fine=pos_points_fine,
            anc_corr_points=anc_corr_points,
            pos_corr_points=pos_corr_points,
            corr_scores=corr_scores,
            transform=T_gt.numpy(),
            score_threshold=0.0,
            anc_feats_fine=anc_feats_fine,
            pos_feats_fine=pos_feats_fine,
            translate=[0,0,50],
            colourmode='umap',
            save_dir=save_dir_ii,
            disable_animation=args.disable_animation,
            non_interactive=args.non_interactive,
        )

        # Best (initial) correspondence from LGR
        visualise_LGR_initial_registration(
            anc_corr_points=best_anc_corr_points,
            pos_corr_points=best_pos_corr_points,
            # transform=T_best_corr.numpy(),
            transform=T_gt.numpy(),
            translate=[0,0,5],
            save_dir=save_dir_ii,
            non_interactive=args.non_interactive,
        )
        
        # Estimated TF
        visualise_registration(
            anc_points_fine=anc_points_fine,
            pos_points_fine=pos_points_fine,
            transform=T_estimated.numpy(),
            # transform=T_gt.numpy(),  # for debugging GT pose
            save_dir=save_dir_ii,
            non_interactive=args.non_interactive,
        )




        
        #       - Plot coarse correspondences
        #       - Plot (some) fine correspondences
        #       - Plot estimated TF
        pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise correspondences from HOTFormerMetricLoc')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--save_dir', type=str, required=False, help='Save visualisations/pcds to directory (creates subdirectories for each model)')
    parser.add_argument('--failures', action='store_true', help='Only visualise metric loc failures')
    parser.add_argument('--idx_list', type=int, nargs='+', default=[], help='Only visualise given list of indices (ordered by val dataloader)')
    parser.add_argument('--disable_animation', action='store_true', help='Disables animations')
    parser.add_argument('--non_interactive', action='store_true', help='Saves visualisations instantly, then closes viewer.')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    if args.save_dir is not None:
        if args.weights is not None and 'weights/' in args.weights:
            model_pathname = (
                f"{os.path.splitext(args.weights.split('weights/')[-1])[0]}"
                f"--{os.path.splitext(os.path.split(args.config)[-1])[0]}"
            )
            args.save_dir = os.path.join(args.save_dir, model_pathname)
            os.makedirs(args.save_dir, exist_ok=True)
        else:
            raise ValueError('Must provide valid path to weights if `--save_dir` is set')
    print('Save dir: {}'.format(args.save_dir))
    print('Failures only: {}'.format(args.failures))
    print('Idx list: {}'.format(args.idx_list))
    if args.failures and len(args.idx_list) > 0:
        print('WARNING: Ignoring `--failures` as `--idx_list` was specified')
    print('Disable animation: {}'.format(args.disable_animation))
    print('Non interactive: {}'.format(args.non_interactive))
    print('Debug mode: {}'.format(args.debug))
    print('')

    set_seed()  # Seed RNG
    main()