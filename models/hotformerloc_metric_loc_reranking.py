"""
HOTFormerLoc class, with metric localisation and re-ranking.
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
from ocnn.octree import Points, Octree

from dataset.augmentation import Normalize
from dataset.coordinate_utils import CoordinateSystem
from models.octree import (
    OctreeT,
    get_octant_centroids_from_points,
    split_and_pad_data,
    unpad_and_concat_data,
)
from models.hotformerloc import HOTFormerLoc
from models.hotformerloc_metric_loc import HOTFormerMetricLoc
from misc.utils import ModelParams
from misc.torch_utils import release_cuda
from models.layers.octformer_layers import MLP
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
)

VIZ = False
SAVE_VIZ_PCL = True
SAVE_DIR = './node_coloring_pcls'

class HOTFormerMetricLocReRanking(HOTFormerMetricLoc):
    def __init__(
        self,
        hotformerloc_global: HOTFormerLoc,
        coarse_feat_refiner: Optional[GeometricTransformer],
        model_params: ModelParams,
        octree_depth: int,
        coarse_idx: int,
        fine_idx: int,
        coarse_feat_embed_dim: Optional[int] = None,
        fine_feat_embed_dim: Optional[int] = None,
        freeze_hotformerloc: bool = False,
        mlp_ratio: float = 2.0,
        quantizer: Optional[CoordinateSystem] = None,
        grad_checkpoint: bool = True,
        return_feats_and_attn_maps: bool = False,
        rerank_mode: Optional[str] = None,
        rerank_geotransformer_refinement: bool = True,
        rerank_num_correspondences: Tuple[int] = (96, 48),
        sort_eigvec: bool = True,
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
            coarse_idx (int): Index corresponding to depth of coarse features, ranging from [0, num_pyramid_levels)
                              (sorted from finest to coarsest). Supports negative indices.
            fine_idx (int): Index corresponding to depth of fine features, ranging from [0, num_pyramid_levels)
                            (sorted from finest to coarsest). Supports negative indices.
            coarse_feat_embed_dim (int): Embedding dim for coarse features (using MLP), set None to disable.
            fine_feat_embed_dim (int): Embedding dim for fine features (using MLP), set None to disable.
            mlp_ratio (int): MLP Ratio in embedding layer.
            freeze_hotformerloc: Freeze HOTFormerLoc backbone layers.
            depth_coarse (int): Octree depth of coarse features (must correspond to depth of OctFormer/HOTFormer blocks)).
            depth_fine (int): Octree depth of fine features (must correspond to depth of OctFormer/HOTFormer blocks)).
            quantizer (CoordinateSystem): Optional quantizer class, used to undo conversion to cylindrical coordinates.
            grad_checkpoint: Use gradient checkpoint to save memory, at cost of extra computation time.
            return_feats_and_attn_maps (bool): Returns intermediate features and attention maps from the backbone.
            rerank_mode (str): Re-ranking method.
            rerank_geotransformer_refinement (bool): Use geotransformer layer to refine features for re-ranking
            rerank_num_correspondences (tuple): Total number of local correspondences for geometric consistency re-ranking, per coarse level.

        Returns:
            model_out (dict): Dict containing outputs from local and global stages
        """
        super().__init__(
            hotformerloc_global=hotformerloc_global,
            coarse_feat_refiner=coarse_feat_refiner,
            model_params=model_params,
            octree_depth=octree_depth,
            coarse_idx=coarse_idx,
            fine_idx=fine_idx,
            coarse_feat_embed_dim=coarse_feat_embed_dim,
            fine_feat_embed_dim=fine_feat_embed_dim,
            freeze_hotformerloc=freeze_hotformerloc,
            mlp_ratio=mlp_ratio,
            quantizer=quantizer,
            grad_checkpoint=grad_checkpoint,
            return_feats_and_attn_maps=return_feats_and_attn_maps,
            **kwargs,
        )
        self.rerank_mode=rerank_mode
        self.rerank_geotransformer_refinement=rerank_geotransformer_refinement
        self.rerank_num_correspondences=rerank_num_correspondences
        self.sort_eigvec = sort_eigvec

        if self.rerank_mode == 'local_hierarchical_gc':
            self.output_dim = sum(self.rerank_num_correspondences)
            self.output_mlp = MLP(self.output_dim, self.output_dim, 1)  # TODO: different hidden size?
            self.sigmoid = nn.Sigmoid()
        else:
            raise NotImplementedError

    def local_hierarchical_gc_rerank(
        self,
        model_out: dict,
        hard_triplets: Tuple,
        shift_and_scale: Tensor,
        points: Points,
        **kwargs,
    ):
        tic_start = time.perf_counter()
        time_dict = {}
        octree: OctreeT = model_out['octree']
        anc_indices, pos_indices, neg_indices = hard_triplets

        # Process each pyramid level
        leading_eigvec_list = []
        for ii, depth_coarse in enumerate(self.depth_coarse):
            # Get local points and features
            local_feats_depth_j = model_out['local'][depth_coarse]
            local_points_depth_j = get_octant_centroids_from_points(points, depth_coarse, self.quantizer)
            if shift_and_scale is not None:
                local_points_depth_j = Normalize.batch_unnormalize_concat(
                    local_points_depth_j, shift_and_scale, octree, depth_coarse
                )
            # Embed coarse feats
            local_feats_depth_j = self.coarse_feat_decoder[ii](local_feats_depth_j)
            # Pad batched tensors and compute padding mask
            local_points_depth_j_padded = split_and_pad_data(
                octree, local_points_depth_j, depth_coarse, fill_value=0.0,
            )
            local_feats_depth_j_padded, local_mask_depth_j_padded = split_and_pad_data(
                octree, local_feats_depth_j, depth_coarse, fill_value=0.0, return_mask=True
            )

            # Separate into anc and nn sets
            anc_points_depth_j_padded = local_points_depth_j_padded[anc_indices]
            B = anc_points_depth_j_padded.size(0)  # recompute B to get correct batch size
            nn_points_depth_j_padded = torch.stack(
                [local_points_depth_j_padded[pos_indices], local_points_depth_j_padded[neg_indices]],
                dim=1,
            )
            anc_feats_depth_j_padded = local_feats_depth_j_padded[anc_indices]
            nn_feats_depth_j_padded = torch.stack(
                [local_feats_depth_j_padded[pos_indices], local_feats_depth_j_padded[neg_indices]],
                dim=1,
            )
            anc_mask_depth_j_padded = local_mask_depth_j_padded[anc_indices]
            nn_mask_depth_j_padded = torch.stack(
                [local_mask_depth_j_padded[pos_indices], local_mask_depth_j_padded[neg_indices]],
                dim=1,
            )

            # TODO: Check anc and nn is in correct format, then forward pass thru
            #       geotrans, and finally create new batched SuperPointMatching class
            
            # # NOTE: Padding does change the output slightly, as it softens the softmax
            # #       output, but is a difference on the order of 0.01-0.1 on average.
            # tic = time.perf_counter()
            # if self.coarse_feat_refiner is not None:
            #     if self.grad_checkpoint and self.training:
            #         anc_feats_coarse_padded, pos_feats_coarse_padded = checkpoint(
            #             self.coarse_feat_refiner,
            #             anc_points_coarse_padded,
            #             pos_points_coarse_padded,
            #             anc_feats_coarse_padded,
            #             pos_feats_coarse_padded,
            #             anc_coarse_mask,
            #             pos_coarse_mask,
            #             use_reentrant=False,
            #         )
            #     else:
            #         anc_feats_coarse_padded, pos_feats_coarse_padded = self.coarse_feat_refiner(
            #             anc_points_coarse_padded,
            #             pos_points_coarse_padded,
            #             anc_feats_coarse_padded,
            #             pos_feats_coarse_padded,
            #             anc_coarse_mask,
            #             pos_coarse_mask,
            #     )
            # anc_feats_coarse_norm_padded = F.normalize(anc_feats_coarse_padded, p=2, dim=1)
            # pos_feats_coarse_norm_padded = F.normalize(pos_feats_coarse_padded, p=2, dim=1)
            # time_dict['geotrans forward'] = time.perf_counter() - tic




        ### CLASSIFIER ###
        #     # Mask out any invalid correspondences (i.e. padding points)
        #     batch_anc_sgv_mask = (batch_anc_final_points == 0).all(dim=-1)  # (B, NN, K)
        #     batch_nn_sgv_mask = (batch_nn_final_points == 0).all(dim=-1)
        #     batch_sgv_mask = torch.logical_not(
        #         torch.logical_or(batch_anc_sgv_mask[..., None], batch_nn_sgv_mask[..., None, :])
        #     )
            
        #     # Compute spectral geometric consistency
        #     leading_eigvec, spatial_consistency_score = batched_sgv_parallel(
        #         batch_anc_final_points,
        #         batch_nn_final_points,
        #         d_thresh=self.geometric_consistency_d_thresh[ii],
        #         return_spatial_consistency=True,
        #         mask=batch_sgv_mask,
        #     )
        #     if self.sort_eigvec:
        #         leading_eigvec, sort_indices = torch.sort(leading_eigvec, dim=-1, descending=True)

        #     # TODO: Scale eigvec? (softmax? or just l2norm?)
        #     leading_eigvec_list.append(leading_eigvec)

        # Concat eigvecs and pass through MLP + sigmoid to get scores
        rerank_features = torch.concat(leading_eigvec_list, dim=-1)
        rerank_scores = self.output_mlp(rerank_features)
        rerank_scores = self.sigmoid(rerank_scores)

        # Create target labels
        targets = torch.zeros_like(rerank_scores)  # [B, 2, 1]
        targets[:, 0] = 1  # set label of positives to 1

        toc = time.perf_counter()
        log_str = f'Re-ranking time: {toc-tic_start:.4f}s'
        logging.debug(log_str)
        return rerank_scores, targets

    def local_hierarchical_gc_rerank_inference(self):
        raise NotImplementedError

    def rerank(self, *args, **kwargs):
        if self.rerank_mode is None:
            return self.hotformerloc_global.rerank(*args, **kwargs)
        elif self.rerank_mode == 'local_hierarchical_gc':
            return self.local_hierarchical_gc_rerank(*args, **kwargs)
        else:
            raise NotImplementedError

    def rerank_inference(self, *args, **kwargs):
        if self.rerank_mode is None:
            return self.hotformerloc_global.rerank_inference(*args, **kwargs)
        elif self.rerank_mode == 'local_hierarchical_gc':
            return self.local_hierarchical_gc_rerank_inference(*args, **kwargs)
        else:
            raise NotImplementedError

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
        # TODO: Fix for hotformermetricloc reranking
        print('Re-ranker: Param count not implemented')
        # if self.hotformerloc_global.reranker is not None:
        #     n_params = sum([param.nelement() for param in self.hotformerloc_global.reranker.parameters()])
        #     print(f'Re-ranker: {type(self.hotformerloc_global.reranker).__name__}\t#parameters: {n_params}')
        # Metric Loc Head
        print('Metric Localisation Head:')
        n_params = sum([param.nelement() for param in self.coarse_feat_decoder.parameters()])
        print(f'  Coarse Feat Decoder: {type(self.coarse_feat_decoder).__name__}\t# parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.fine_feat_decoder.parameters()])
        print(f'  Fine Feat Decoder: {type(self.fine_feat_decoder).__name__}\t# parameters: {n_params}')
        if self.coarse_feat_refiner is not None:
            n_params = sum([param.nelement() for param in self.coarse_feat_refiner.parameters()])
            print(f'  Coarse Feat Refiner: {type(self.coarse_feat_refiner).__name__}\t# parameters: {n_params}')
