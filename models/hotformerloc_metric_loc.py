"""
HOTFormerLoc class.
Author: Ethan Griffiths
CSIRO Data61

Code adapted from OctFormer: Octree-based Transformers for 3D Point Clouds
by Peng-Shuai Wang.
"""
import time
import logging
from typing import Optional, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from ocnn.octree import Points, Octree

from dataset.augmentation import Normalize
from dataset.coordinate_utils import CoordinateSystem
from models.octree import get_octant_centroids_from_points, split_and_pad_data, unpad_and_concat_data
from models.hotformerloc import HOTFormerLoc
from misc.utils import ModelParams
from misc.torch_utils import release_cuda
from misc.poses import invert_pose
from models.layers.local_global_registration import LocalGlobalRegistration
from models.layers.octformer_layers import MLP
from models.reranking_utils import sgv_parallel
from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointTargetGenerator,
    SuperPointMatching,
)
from geotransformer.utils.visualization import (
    draw_point_to_node,
    draw_node_correspondences,
    get_colors_with_tsne
)

VIZ = False
SAVE_VIZ_PCL = True
SAVE_DIR = './node_coloring_pcls'

class HOTFormerMetricLoc(torch.nn.Module):
    def __init__(
        self,
        hotformerloc_global: HOTFormerLoc,
        coarse_feat_refiner: Optional[nn.ModuleList],
        model_params: ModelParams,
        octree_depth: int,
        coarse_idx: Tuple[int],
        fine_idx: int,
        coarse_feat_embed_dim: Optional[Tuple[int]] = None,
        fine_feat_embed_dim: Optional[int] = None,
        freeze_hotformerloc: bool = False,
        mlp_ratio: float = 2.0,
        quantizer: Optional[CoordinateSystem] = None,
        grad_checkpoint: bool = True,
        return_feats_and_attn_maps: bool = False,
        **kwargs,
    ):
        """
        Class for HOTFormerLoc-based metric localisation, with coarse-to-fine
        registration inspired by GeoTransformer.

        Args:
            hotformerloc_global (nn.Module): HOTFormerLoc instance for extracting local features and global descriptor for place rec.
            coarse_feat_refiner (nn.Module): GeoTransformer (or other) instance for refining coarse features and correspondences.
            model_params (ModelParams): Model parameters instance.
            octree_depth (int): Octree depth (must be fixed for this instance of the model).
            coarse_idx (tuple): List of indices corresponding to depth of coarse features, ranging from [0, num_pyramid_levels)
                                (sorted from finest to coarsest). Supports negative indices.
            fine_idx (int): Index corresponding to depth of fine features, ranging from [0, num_pyramid_levels)
                            (sorted from finest to coarsest). Supports negative indices.
            coarse_feat_embed_dim (int): Embedding dim for coarse features (using MLP), set None to disable
            fine_feat_embed_dim (int): Embedding dim for fine features (using MLP), set None to disable
            mlp_ratio: MLP Ratio in embedding layer
            freeze_hotformerloc: Freeze HOTFormerLoc backbone layers
            depth_coarse (int): Octree depth of coarse features (must correspond to depth of OctFormer/HOTFormer blocks))
            depth_fine (int): Octree depth of fine features (must correspond to depth of OctFormer/HOTFormer blocks))
            quantizer (CoordinateSystem): Optional quantizer class, used to undo conversion to cylindrical coordinates
            grad_checkpoint: Use gradient checkpoint to save memory, at cost of extra computation time.
            return_feats_and_attn_maps (bool): Returns intermediate features and attention maps from the backbone.

        Returns:
            model_out (dict): Dict containing outputs from local and global stages

        """
        super().__init__()
        self.hotformerloc_global = hotformerloc_global
        self.coarse_feat_refiner = coarse_feat_refiner
        self.model_params = model_params
        self.coarse_idx = coarse_idx
        self.fine_idx = fine_idx
        self.coarse_feat_embed_dim = coarse_feat_embed_dim
        self.fine_feat_embed_dim = fine_feat_embed_dim
        self.mlp_ratio = mlp_ratio
        self.freeze_hotformerloc = freeze_hotformerloc
        self.octree_depth = octree_depth
        self.compute_depth_from_idx()
        self.get_input_dim()
        self.quantizer = quantizer
        self.grad_checkpoint = grad_checkpoint
        self.return_feats_and_attn_maps = return_feats_and_attn_maps
        self.point_padding = 1e10
        self._benchmark = False
        for depth_coarse_ii in self.depth_coarse:
            if depth_coarse_ii >= self.depth_fine:
                err_str = ('Coarse feature depth in octree must be less than fine'
                           ' feature depth, check idx parameters.')
                raise ValueError(err_str)
            if depth_coarse_ii < 1:
                err_str = 'Select a valid feature depth (minimum 1)'
                raise ValueError(err_str)

        if self.freeze_hotformerloc:
            for param in self.hotformerloc_global.parameters():
                param.requires_grad = False

        self.coarse_feat_decoder = nn.ModuleList()
        for ii, coarse_feat_input_dim in enumerate(self.coarse_feat_input_dim):
            self.coarse_feat_decoder.append(
                MLP(
                    coarse_feat_input_dim,
                    int(coarse_feat_input_dim * self.mlp_ratio),
                    self.coarse_feat_embed_dim[ii],
                ) if self.coarse_feat_embed_dim is not None else nn.Identity()
            )
        self.fine_feat_decoder = MLP(
            self.fine_feat_input_dim,
            int(self.fine_feat_input_dim * self.mlp_ratio),
            self.fine_feat_embed_dim,
        ) if self.fine_feat_embed_dim is not None else nn.Identity()

        self.coarse_target = SuperPointTargetGenerator(
            model_params.coarse_matching.num_targets,
            model_params.coarse_matching.overlap_threshold,
        )

        self.coarse_matching = SuperPointMatching(
            model_params.coarse_matching.num_correspondences,
            model_params.coarse_matching.dual_normalization,
        )
        self.num_points_in_patch = model_params.coarse_matching.num_points_in_patch
        self.matching_radius = model_params.coarse_matching.ground_truth_matching_radius

        self.fine_matching = LocalGlobalRegistration(
            model_params.fine_matching.topk,
            model_params.fine_matching.acceptance_radius,
            mutual=model_params.fine_matching.mutual,
            confidence_threshold=model_params.fine_matching.confidence_threshold,
            use_dustbin=model_params.fine_matching.use_dustbin,
            use_global_score=model_params.fine_matching.use_global_score,
            correspondence_threshold=model_params.fine_matching.correspondence_threshold,
            correspondence_limit=model_params.fine_matching.correspondence_limit,
            num_refinement_steps=model_params.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(
            model_params.fine_matching.num_sinkhorn_iterations
        )

    def benchmark(self, mode: bool = True):
        """
        Set benchmarking mode (disables functions only needed for training/evaluation)
        """
        if not isinstance(mode, bool):
            raise ValueError('benchmark mode is expected to be boolean')
        if self.training and mode:
            raise ValueError('model must be in eval mode to enable benchmark mode')
        self._benchmark = mode
        return self

    def compute_depth_from_idx(self):
        """
        Determines the octree depth of coarse and fine features based on the
        input octree depth and HOTFormerLoc parameters.
        """
        num_downsamples = (self.hotformerloc_global.backbone.backbone.stem_down
                           if self.hotformerloc_global.backbone.backbone.downsample_input_embeddings
                           else 0)
        num_stages = self.hotformerloc_global.backbone.backbone.num_stages
        depth_start = self.octree_depth - num_downsamples
        self.depth_coarse = []
        for coarse_idx in self.coarse_idx:
            if coarse_idx >= 0:
                depth_coarse = depth_start - coarse_idx
            else:  # neg idx
                depth_coarse = depth_start - num_stages - coarse_idx
            self.depth_coarse.append(depth_coarse)
        if self.fine_idx >= 0:
            self.depth_fine = depth_start - self.fine_idx
        else:
            self.depth_fine = depth_start - num_stages - self.fine_idx

    def get_input_dim(self):
        """
        Determing coarse and fine feature input dimensions based on HOTFormerLoc
        parameters.
        """
        channels = list(self.hotformerloc_global.backbone.backbone.channels)
        num_octf_levels = self.hotformerloc_global.backbone.backbone.num_octf_levels
        num_pyramid_levels = self.hotformerloc_global.backbone.backbone.num_pyramid_levels
        if len(channels[num_octf_levels:]) == 1:
            channels[num_octf_levels:] = channels[num_octf_levels:] * num_pyramid_levels
        self.coarse_feat_input_dim = []
        for coarse_idx in self.coarse_idx:
            self.coarse_feat_input_dim.append(channels[coarse_idx])
        self.fine_feat_input_dim = channels[self.fine_idx]

    def forward(self, batch, global_only=False, **kwargs) -> List[dict]:
        """
        Batched implementation not finished. Currently processes the coarse features
        in batched mode, then just loops through each batch elem for coarse matching
        and fine registration. Returns a list of output dicts with len == batch_size.
        """
        # NOTE: anc = anchor, pos = positive
        tic_start = time.perf_counter()
        time_dict = {}

        # 1. Extract global and local descriptors with HOTFormerLoc
        if global_only:
            global_out = self.hotformerloc_global(batch)
            time_dict['global backbone forward'] = time.perf_counter() - tic_start
            self.log_time_dict(time_dict)
            return global_out

        anc_octree: Octree = batch['anc_batch']['octree']
        pos_octree: Octree = batch['pos_batch']['octree']
        anc_points: Points = batch['anc_batch']['points']
        pos_points: Points = batch['pos_batch']['points']

        # Get coarse and fine feats and points
        if len(self.depth_coarse) > 1:
            # raise NotImplementedError('Hierarchical coarse feat refinement not yet implemented')
            pass
        # NOTE: Temporarily set the coarse depth manually, until multi-stage coarse refinement is implemented
        depth_coarse = self.depth_coarse[0]
        coarse_ii = 0
        
        time_dict['local backbone forward'] = 0.0
        if 'anc_local_feats' in batch:
            # If pre-computed, skip forward pass
            anc_feats_coarse = batch['anc_local_feats'][depth_coarse]
            anc_feats_fine = batch['anc_local_feats'][self.depth_fine]
        else:
            # TODO: Process anchor and positive batch in single forward pass
            tic = time.perf_counter()
            anc_global_out = self.hotformerloc_global(batch['anc_batch'])
            time_dict['local backbone forward'] += time.perf_counter() - tic
            anc_feats_coarse = anc_global_out['local'][depth_coarse]
            anc_feats_fine = anc_global_out['local'][self.depth_fine]
        if 'pos_local_feats' in batch:
            pos_feats_coarse = batch['pos_local_feats'][depth_coarse]
            pos_feats_fine = batch['pos_local_feats'][self.depth_fine]
        else:
            tic = time.perf_counter()
            pos_global_out = self.hotformerloc_global(batch['pos_batch'])
            time_dict['local backbone forward'] += time.perf_counter() - tic
            pos_feats_coarse = pos_global_out['local'][depth_coarse]
            pos_feats_fine = pos_global_out['local'][self.depth_fine]

        # Embed coarse and fine feats
        tic = time.perf_counter()
        anc_feats_coarse = self.coarse_feat_decoder[coarse_ii](anc_feats_coarse)
        anc_feats_fine = self.fine_feat_decoder(anc_feats_fine)
        pos_feats_coarse = self.coarse_feat_decoder[coarse_ii](pos_feats_coarse)
        pos_feats_fine = self.fine_feat_decoder(pos_feats_fine)
        time_dict['feat decoder'] = time.perf_counter() - tic

        # Get accurate centroids for octants (instead of naively using octant centres)
        tic = time.perf_counter()
        anc_points_coarse = get_octant_centroids_from_points(anc_points, depth_coarse, self.quantizer)
        pos_points_coarse = get_octant_centroids_from_points(pos_points, depth_coarse, self.quantizer)
        anc_points_fine = get_octant_centroids_from_points(anc_points, self.depth_fine, self.quantizer)
        pos_points_fine = get_octant_centroids_from_points(pos_points, self.depth_fine, self.quantizer)
        time_dict['compute centroids'] = time.perf_counter() - tic

        output_dicts = [{} for _ in range(anc_octree.batch_size)]

        ########################################################################
        # DEBUGGING - VISUALISING COARSE AND FINE POINTS
        ########################################################################
        # from misc.point_clouds import plot_points
        # from misc.torch_utils import release_cuda
        # plot_batch_id = 0
        # plot_points(release_cuda(anc_points_coarse[anc_octree.batch_id(depth_coarse, nempty=True) == plot_batch_id], to_numpy=True))
        # plot_points(release_cuda(pos_points_coarse[pos_octree.batch_id(depth_coarse, nempty=True) == plot_batch_id], to_numpy=True))
        # plot_points(release_cuda(anc_points_fine[anc_octree.batch_id(depth_fine, nempty=True) == plot_batch_id], to_numpy=True))
        # plot_points(release_cuda(pos_points_fine[pos_octree.batch_id(depth_fine, nempty=True) == plot_batch_id], to_numpy=True))
        ########################################################################

        # 2. Separate anchor and positive feats/points, and process in GeoTrans
        # TODO: Add check here for eval, as in eval, we will pass one anchor and N positives
        if anc_octree.batch_size == 1 and pos_octree.batch_size > 1:
            raise NotImplementedError('Eval not implemented')

        # Undo normalization so points are in metric scale
        tic = time.perf_counter()
        if batch['anc_shift_and_scale'] is not None:
            anc_points_coarse = self.batch_unnormalize(
                anc_points_coarse, batch['anc_shift_and_scale'], anc_octree, depth_coarse
            )
            anc_points_fine = self.batch_unnormalize(
                anc_points_fine, batch['anc_shift_and_scale'], anc_octree, self.depth_fine
            )
        if batch['pos_shift_and_scale'] is not None:
            pos_points_coarse = self.batch_unnormalize(
                pos_points_coarse, batch['pos_shift_and_scale'], pos_octree, depth_coarse
            )
            pos_points_fine = self.batch_unnormalize(
                pos_points_fine, batch['pos_shift_and_scale'], pos_octree, self.depth_fine
            )
        time_dict['unnormalize'] = time.perf_counter() - tic

        # Pad batched tensors and create attn mask
        #   (pad points with large value to prevent interactions with distance embedding)
        anc_points_coarse_padded = split_and_pad_data(
            anc_octree, anc_points_coarse, depth_coarse, fill_value=self.point_padding
        )
        pos_points_coarse_padded = split_and_pad_data(
            pos_octree, pos_points_coarse, depth_coarse, fill_value=self.point_padding
        )
        anc_feats_coarse_padded, anc_coarse_mask = split_and_pad_data(
            anc_octree, anc_feats_coarse, depth_coarse, fill_value=0., return_mask=True
        )
        pos_feats_coarse_padded, pos_coarse_mask = split_and_pad_data(
            pos_octree, pos_feats_coarse, depth_coarse, fill_value=0., return_mask=True
        )

        # NOTE: Padding does change the output slightly, as it softens the softmax
        #       output, but is a difference on the order of 0.01-0.1 on average.
        if self.coarse_feat_refiner[coarse_ii] is not None:
            tic = time.perf_counter()
            if self.grad_checkpoint and self.training:
                anc_feats_coarse_padded, pos_feats_coarse_padded = checkpoint(
                    self.coarse_feat_refiner[coarse_ii],
                    anc_points_coarse_padded,
                    pos_points_coarse_padded,
                    anc_feats_coarse_padded,
                    pos_feats_coarse_padded,
                    anc_coarse_mask,
                    pos_coarse_mask,
                    use_reentrant=False,
                )
            else:
                anc_feats_coarse_padded, pos_feats_coarse_padded = self.coarse_feat_refiner[coarse_ii](
                    anc_points_coarse_padded,
                    pos_points_coarse_padded,
                    anc_feats_coarse_padded,
                    pos_feats_coarse_padded,
                    anc_coarse_mask,
                    pos_coarse_mask,
                )
            time_dict['geotrans forward'] = time.perf_counter() - tic
        anc_feats_coarse_norm_padded = F.normalize(anc_feats_coarse_padded, p=2, dim=1)
        pos_feats_coarse_norm_padded = F.normalize(pos_feats_coarse_padded, p=2, dim=1)

        # Convert feats back to concatenated form
        anc_feats_coarse_norm = unpad_and_concat_data(
            anc_octree, anc_feats_coarse_norm_padded, depth_coarse
        )
        pos_feats_coarse_norm = unpad_and_concat_data(
            pos_octree, pos_feats_coarse_norm_padded, depth_coarse
        )

        time_dict['coarse matching'] = 0.0
        time_dict['optimal transport'] = 0.0
        time_dict['fine matching'] = 0.0
        tic_lgr_start = time.perf_counter()
        # TODO: Convert coarse matching to work in batches (may require zero padding or selecting k coarse points)
        for batch_idx in range(pos_octree.batch_size):
            anc_batch_mask_coarse = anc_octree.batch_id(depth_coarse, nempty=True) == batch_idx
            pos_batch_mask_coarse = pos_octree.batch_id(depth_coarse, nempty=True) == batch_idx
            anc_batch_mask_fine = anc_octree.batch_id(self.depth_fine, nempty=True) == batch_idx
            pos_batch_mask_fine = pos_octree.batch_id(self.depth_fine, nempty=True) == batch_idx
            anc_points_coarse_ii = anc_points_coarse[anc_batch_mask_coarse]
            pos_points_coarse_ii = pos_points_coarse[pos_batch_mask_coarse]
            anc_feats_coarse_norm_ii = anc_feats_coarse_norm[anc_batch_mask_coarse]
            pos_feats_coarse_norm_ii = pos_feats_coarse_norm[pos_batch_mask_coarse]
            anc_feats_coarse_pre_refinement_ii = anc_feats_coarse[anc_batch_mask_coarse]
            pos_feats_coarse_pre_refinement_ii = pos_feats_coarse[pos_batch_mask_coarse]
            anc_points_fine_ii = anc_points_fine[anc_batch_mask_fine]
            pos_points_fine_ii = pos_points_fine[pos_batch_mask_fine]
            anc_feats_fine_ii = anc_feats_fine[anc_batch_mask_fine]
            pos_feats_fine_ii = pos_feats_fine[pos_batch_mask_fine]
            transform_ii = batch['transform'][batch_idx]

            # Separate dict for each batch item
            output_dicts[batch_idx]['anc_feats_coarse'] = anc_feats_coarse_norm_ii
            output_dicts[batch_idx]['pos_feats_coarse'] = pos_feats_coarse_norm_ii
            output_dicts[batch_idx]['anc_feats_coarse_pre_refinement'] = anc_feats_coarse_pre_refinement_ii
            output_dicts[batch_idx]['pos_feats_coarse_pre_refinement'] = pos_feats_coarse_pre_refinement_ii
            output_dicts[batch_idx]['anc_feats_fine'] = anc_feats_fine_ii
            output_dicts[batch_idx]['pos_feats_fine'] = pos_feats_fine_ii
            output_dicts[batch_idx]['anc_points_coarse'] = anc_points_coarse_ii
            output_dicts[batch_idx]['pos_points_coarse'] = pos_points_coarse_ii
            output_dicts[batch_idx]['anc_points_fine'] = anc_points_fine_ii
            output_dicts[batch_idx]['pos_points_fine'] = pos_points_fine_ii

            # 3. Generate ground truth node correspondences
            # NOTE: step 3 should be achievable just from octree indices alone
            anc_point_to_node_ii, anc_node_masks_ii, anc_node_knn_indices_ii, anc_node_knn_masks_ii = point_to_node_partition(
                anc_points_fine_ii, anc_points_coarse_ii, self.num_points_in_patch
            )
            pos_point_to_node_ii, pos_node_masks_ii, pos_node_knn_indices_ii, pos_node_knn_masks_ii = point_to_node_partition(
                pos_points_fine_ii, pos_points_coarse_ii, self.num_points_in_patch
            )
            output_dicts[batch_idx]['anc_point_to_node'] = anc_point_to_node_ii
            output_dicts[batch_idx]['pos_point_to_node'] = pos_point_to_node_ii

            if VIZ:
                save_filename = f'{SAVE_DIR}/superpoints-{batch_idx}' if SAVE_VIZ_PCL else None
                draw_point_to_node(release_cuda(anc_points_fine_ii, to_numpy=True),
                                   release_cuda(anc_points_coarse_ii, to_numpy=True),
                                   release_cuda(anc_point_to_node_ii, to_numpy=True),
                                   save_basepath=save_filename,
                                   viz=not SAVE_VIZ_PCL)

            anc_padded_points_fine_ii = torch.cat([anc_points_fine_ii, torch.zeros_like(anc_points_fine_ii[:1])], dim=0)
            pos_padded_points_fine_ii = torch.cat([pos_points_fine_ii, torch.zeros_like(pos_points_fine_ii[:1])], dim=0)
            anc_node_knn_points_ii = index_select(anc_padded_points_fine_ii, anc_node_knn_indices_ii, dim=0)
            pos_node_knn_points_ii = index_select(pos_padded_points_fine_ii, pos_node_knn_indices_ii, dim=0)

            if not self._benchmark:
                gt_node_corr_indices_ii, gt_node_corr_overlaps_ii = get_node_correspondences(
                    anc_points_coarse_ii,
                    pos_points_coarse_ii,
                    anc_node_knn_points_ii,
                    pos_node_knn_points_ii,
                    invert_pose(transform_ii),  # NOTE: GeoTrans expects transform from src (pos) to ref (anc)
                    self.matching_radius,
                    ref_masks=anc_node_masks_ii,
                    src_masks=pos_node_masks_ii,
                    ref_knn_masks=anc_node_knn_masks_ii,
                    src_knn_masks=pos_node_knn_masks_ii,
                )
                # if VIZ:
                #     draw_node_correspondences(
                #         release_cuda(anc_points_fine_ii), release_cuda(anc_points_coarse_ii),
                #         release_cuda(anc_point_to_node_ii), release_cuda(pos_points_fine_ii),
                #         release_cuda(pos_points_coarse_ii), release_cuda(pos_point_to_node_ii),
                #         'pos', offsets=(0, 200, 0),
                #     )

                output_dicts[batch_idx]['gt_node_corr_indices'] = gt_node_corr_indices_ii
                output_dicts[batch_idx]['gt_node_corr_overlaps'] = gt_node_corr_overlaps_ii

            # 4. Select topk nearest node correspondences
            with torch.no_grad():
                tic = time.perf_counter()
                anc_node_corr_indices_ii, pos_node_corr_indices_ii, node_corr_scores_ii = self.coarse_matching(
                    anc_feats_coarse_norm_ii, pos_feats_coarse_norm_ii, anc_node_masks_ii, pos_node_masks_ii
                )
                time_dict['coarse matching'] += time.perf_counter() - tic

                output_dicts[batch_idx]['anc_node_corr_indices'] = anc_node_corr_indices_ii
                output_dicts[batch_idx]['pos_node_corr_indices'] = pos_node_corr_indices_ii

                # 4.1 Randomly select ground truth node correspondences during training
                if not self._benchmark:
                    if self.training:
                        anc_node_corr_indices_ii, pos_node_corr_indices_ii, node_corr_scores_ii = self.coarse_target(
                            gt_node_corr_indices_ii, gt_node_corr_overlaps_ii
                        )
                    if len(node_corr_scores_ii) == 0:
                        log_str = (
                            'No ground truth node correspondences found -- check '
                            '`ground_truth_matching_radius`, `overlap_threshold`, and '
                            'coarse/fine resolutions (likely too coarse) -- '
                            'this will cause NaNs to be logged'
                        )
                        logging.warning(log_str)

        # # TODO: CONTINUE LOOP FROM HERE, BUT THINK ABOUT HOW TO HANDLE EVAL

        # # TODO: Implement SpectralGV re-ranking based on refined coarse correspondences 
        # if not self.training:
        #     with torch.no_grad():
        #         pass

        # NOTE: Temporarily just placing this inside the previous for loop, just
        #       to get training running
        # # 5 Generate batched node points & feats
        # for batch_idx in range(pos_octree.batch_size):
            anc_node_corr_knn_indices_ii = anc_node_knn_indices_ii[anc_node_corr_indices_ii]  # (P, K)
            pos_node_corr_knn_indices_ii = pos_node_knn_indices_ii[pos_node_corr_indices_ii]  # (P, K)
            anc_node_corr_knn_masks_ii = anc_node_knn_masks_ii[anc_node_corr_indices_ii]  # (P, K)
            pos_node_corr_knn_masks_ii = pos_node_knn_masks_ii[pos_node_corr_indices_ii]  # (P, K)
            anc_node_corr_knn_points_ii = anc_node_knn_points_ii[anc_node_corr_indices_ii]  # (P, K, 3)
            pos_node_corr_knn_points_ii = pos_node_knn_points_ii[pos_node_corr_indices_ii]  # (P, K, 3)

            anc_padded_feats_fine_ii = torch.cat([anc_feats_fine_ii, torch.zeros_like(anc_feats_fine_ii[:1])], dim=0)
            pos_padded_feats_fine_ii = torch.cat([pos_feats_fine_ii, torch.zeros_like(pos_feats_fine_ii[:1])], dim=0)
            anc_node_corr_knn_feats_ii = index_select(anc_padded_feats_fine_ii, anc_node_corr_knn_indices_ii, dim=0)  # (P, K, C)
            pos_node_corr_knn_feats_ii = index_select(pos_padded_feats_fine_ii, pos_node_corr_knn_indices_ii, dim=0)  # (P, K, C)

            output_dicts[batch_idx]['anc_node_corr_knn_points'] = anc_node_corr_knn_points_ii
            output_dicts[batch_idx]['pos_node_corr_knn_points'] = pos_node_corr_knn_points_ii
            output_dicts[batch_idx]['anc_node_corr_knn_masks'] = anc_node_corr_knn_masks_ii
            output_dicts[batch_idx]['pos_node_corr_knn_masks'] = pos_node_corr_knn_masks_ii

            # 6. Optimal transport
            tic = time.perf_counter()
            matching_scores_ii = torch.einsum('bnd,bmd->bnm', anc_node_corr_knn_feats_ii, pos_node_corr_knn_feats_ii)  # (P, K, K)
            matching_scores_ii = matching_scores_ii / anc_feats_fine_ii.shape[-1] ** 0.5
            if self.grad_checkpoint and self.training:
                matching_scores_ii = checkpoint(
                    self.optimal_transport,
                    matching_scores_ii,
                    anc_node_corr_knn_masks_ii,
                    pos_node_corr_knn_masks_ii,
                    use_reentrant=False,
                )
            else:
                matching_scores_ii = self.optimal_transport(
                    matching_scores_ii,
                    anc_node_corr_knn_masks_ii,
                    pos_node_corr_knn_masks_ii,
                )
            time_dict['optimal transport'] += time.perf_counter() - tic

            output_dicts[batch_idx]['matching_scores'] = matching_scores_ii

            # 7. Generate final correspondences during testing
            with torch.no_grad():
                if not self.fine_matching.use_dustbin:
                    matching_scores_ii = matching_scores_ii[:, :-1, :-1]

                # NOTE: estimated transform is from pos to anc
                tic = time.perf_counter()
                (
                    anc_corr_points_ii,
                    pos_corr_points_ii,
                    corr_scores_ii,
                    estimated_transform_ii,
                    num_corr_points_ii,
                    best_anc_corr_points_ii,
                    best_pos_corr_points_ii,
                    best_corr_scores_ii,
                    best_transform_ii,
                ) = self.fine_matching(
                    anc_node_corr_knn_points_ii,
                    pos_node_corr_knn_points_ii,
                    anc_node_corr_knn_masks_ii,
                    pos_node_corr_knn_masks_ii,
                    matching_scores_ii,
                    node_corr_scores_ii,
                )
                time_dict['fine matching'] += time.perf_counter() - tic

                output_dicts[batch_idx]['anc_corr_points'] = anc_corr_points_ii
                output_dicts[batch_idx]['pos_corr_points'] = pos_corr_points_ii
                output_dicts[batch_idx]['corr_scores'] = corr_scores_ii
                output_dicts[batch_idx]['num_corr_points'] = num_corr_points_ii
                output_dicts[batch_idx]['best_anc_corr_points'] = best_anc_corr_points_ii
                output_dicts[batch_idx]['best_pos_corr_points'] = best_pos_corr_points_ii
                output_dicts[batch_idx]['best_corr_scores'] = best_corr_scores_ii
                # Ensure estimated transform is from anc to pos, not vice-versa
                output_dicts[batch_idx]['estimated_transform'] = invert_pose(estimated_transform_ii)
                if best_transform_ii is not None:
                    best_transform_ii = invert_pose(best_transform_ii)
                output_dicts[batch_idx]['best_corr_transform'] = best_transform_ii

        toc = time.perf_counter()
        time_dict['local global reg (whole loop)'] = toc - tic_lgr_start
        time_dict['TOTAL'] = toc - tic_start
        self.log_time_dict(time_dict)
        return output_dicts

    def batch_unnormalize(
        self,
        points: torch.Tensor,
        batch_shift_and_scale: torch.Tensor,
        octree: Octree,
        depth: int,
    ):
        """
        Undo normalization for a batch of points. 
        """
        assert points.size(0) == octree.batch_id(depth, nempty=True).size(0), (
            f"Points must correspond to the octree at depth {depth}"
        )
        # TODO: Implement vectorised form of this
        for batch_idx in range(octree.batch_size):
            batch_mask = octree.batch_id(depth, nempty=True) == batch_idx
            points[batch_mask] = Normalize.unnormalize(
                points[batch_mask],
                batch_shift_and_scale[batch_idx],
            )
        return points

    def log_time_dict(self, time_dict: dict, initial_str: str = 'Model forward pass:  '):
        time_str = '' + initial_str
        for name, process_time in time_dict.items():
            if name == 'TOTAL':
                time_str += f'TOTAL: {process_time:.4f}s'
            else:
                time_str += f'{name}: {process_time:.4f}s,  '
        logging.debug(time_str)

    def rerank(self, *args, **kwargs):
        return self.hotformerloc_global.rerank(*args, **kwargs)

    def rerank_inference(self, *args, **kwargs):
        return self.hotformerloc_global.rerank_inference(*args, **kwargs)

    def sgv_rerank_inference(
        self,
        model_out: dict,
        shift_and_scale: Tensor,
        batch: dict,
        feat_type: str = 'coarse',
        **kwargs,
    ):
        """
        Perform sgv re-ranking of query and N candidates. Note that only 'local'
        keys are needed from HOTFormerLoc outputs.

        Args:
            model_out (dict): HOTFormerLoc output for entire (inference) batch. Assums first batch item is query, and all others are candidates.
            shift_and_scale (Tensor): (B, 4) tensor containing normalization parameters
            batch (dict): Batch containing `octree` and `points` objects

        Returns:
            rerank_dict (dict)
        """
        assert feat_type in ('coarse', 'fine')
        tic_start = time.perf_counter()
        time_dict = {}
        octree: Octree = batch['octree']
        points: Points = batch['points']
        anc_idx = 0
        NN = octree.batch_size - 1  # first elem is query

        coarse_feat_ii = 0
        if feat_type == 'coarse':
            # Use just the coarsest level of feats for SGV
            depth_j = self.depth_coarse[coarse_feat_ii]
            sgv_d_thresh = 0.4
        elif feat_type == 'fine':
            depth_j = self.depth_fine
            sgv_d_thresh = 0.4  # SGV dist threshold used in the paper
        # print(f'sgv_d_thresh={sgv_d_thresh}')

        # Get local points and features
        if isinstance(model_out['local'], dict):  # standard HOTFloc output
            local_feats = model_out['local'][depth_j]
        elif isinstance(model_out['local'], list):  # inference output, nested list of local feats from query and NNs
            local_feats_list = [feat_dict[depth_j] for feat_dict in model_out['local']]
            local_feats = torch.concat(local_feats_list, dim=0)
            assert local_feats.size(0) == octree.batch_nnum_nempty[depth_j].sum(), 'Octree does not match local feats'
        local_points = get_octant_centroids_from_points(points, depth_j, self.quantizer)
        if shift_and_scale is not None:
            local_points = Normalize.batch_unnormalize_concat(
                local_points, shift_and_scale, octree, depth_j
            )
        # Split batch into list
        batch_lengths = octree.batch_nnum_nempty[depth_j].tolist()
        local_points_list = local_points.split(batch_lengths)
        local_feats_list = local_feats.split(batch_lengths)
            
        # Process SGV on each query/nn pair individually
        tic = time.perf_counter()
        leading_eigvec_list = []
        fitness_list = []
        for nn_idx in range(NN):
            # Collect points & feats for pair, and trim num points to the smallest one
            # NOTE: This is the same method used in SGV for LoGG3D-Net
            anc_points = local_points_list[anc_idx]
            nn_points = local_points_list[nn_idx + 1]
            anc_feats = local_feats_list[anc_idx]
            nn_feats = local_feats_list[nn_idx + 1]

            min_num_feat = min(len(anc_points),len(nn_points))
            anc_points = anc_points[:min_num_feat]
            nn_points = nn_points[:min_num_feat]
            anc_feats = anc_feats[:min_num_feat]
            nn_feats = nn_feats[:min_num_feat]
            
            # Compute spectral geometric consistency
            leading_eigvec, spatial_consistency_score = sgv_parallel(
                anc_points[None, ...],
                nn_points[None, ...],
                anc_feats[None, ...],
                nn_feats[None, ...],
                d_thresh=sgv_d_thresh,
                return_spatial_consistency=True,
            )
            leading_eigvec_list.append(leading_eigvec[0])
            fitness_list.append(spatial_consistency_score.item())
        time_dict[f'sgv {coarse_feat_ii}'] = time.perf_counter() - tic

        toc = time.perf_counter()
        time_dict['TOTAL'] = toc - tic_start
        self.log_time_dict(time_dict, initial_str='Re-ranking:  ')
        return {'scores': torch.tensor(fitness_list),
                'eigvec_list': leading_eigvec_list}

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Set of parameters that should not use weight decay."""
        return {'rpe_table', 'rt_init_token', 'rt_cls_token'}
        
    def print_info(self):
        print('Model class: HOTFormerMetricLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        # Backbone
        n_params = sum([param.nelement() for param in self.hotformerloc_global.backbone.parameters()])
        print(f'Backbone: {type(self.hotformerloc_global.backbone).__name__}\t# parameters: {n_params}')
        base_model = self.hotformerloc_global.backbone.backbone
        n_params = sum([param.nelement() for param in base_model.patch_embed.parameters()])
        print(f"  ConvEmbed:\t# parameters: {n_params}")
        n_params = sum([param.nelement() for param in base_model.octf_stage.parameters()])
        n_params += sum([param.nelement() for param in base_model.downsample.parameters()])
        print(f"  OctF Layers:\t# parameters: {n_params}")
        n_params = sum([param.nelement() for param in base_model.hotf_stage.parameters()])
        print(f"  HOTF Layers:\t# parameters: {n_params}")
        # Pooling
        n_params = sum([param.nelement() for param in self.hotformerloc_global.pooling.parameters()])
        print(f'Pooling method: {self.hotformerloc_global.pooling.pool_method}\t# parameters: {n_params}')
        print('  # channels from the backbone: {}'.format(self.hotformerloc_global.pooling.in_dim))
        print('  # output channels : {}'.format(self.hotformerloc_global.pooling.output_dim))
        print(f'  Embedding normalization: {self.hotformerloc_global.normalize_embeddings}')
        # Reranker
        if self.hotformerloc_global.reranker is not None:
            n_params = sum([param.nelement() for param in self.hotformerloc_global.reranker.parameters()])
            print(f'Re-ranker: {type(self.hotformerloc_global.reranker).__name__}\t#parameters: {n_params}')
        # Metric Loc Head
        print('Metric Localisation Head:')
        n_params = sum([param.nelement() for param in self.coarse_feat_decoder.parameters()])
        print(f'  Coarse Feat Decoder: {type(self.coarse_feat_decoder).__name__}\t# parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.fine_feat_decoder.parameters()])
        print(f'  Fine Feat Decoder: {type(self.fine_feat_decoder).__name__}\t# parameters: {n_params}')
        if self.coarse_feat_refiner is not None:
            n_params = sum([param.nelement() for param in self.coarse_feat_refiner.parameters()])
            print(f'  Coarse Feat Refiner: {type(self.coarse_feat_refiner).__name__}\t# parameters: {n_params}')
