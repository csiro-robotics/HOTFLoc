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
from torch.utils.checkpoint import checkpoint
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
from models.layers.octformer_layers import MLP
from models.reranking_utils import batched_sgv_parallel, mutual_topk_correspondences
from misc.utils import ModelParams
from misc.torch_utils import release_cuda, min_max_normalize
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
)


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
        rerank_indices: Tuple[int] = (-1, -2),
        rerank_feat_embed_dim: Optional[Tuple[int]] = None,
        geometric_consistency_d_thresh: Tuple[float] = (0.6, 0.6),
        rerank_geotransformer_refinement: bool = True,
        rerank_num_correspondences: Tuple[int] = (128, 256),
        rerank_sort_eigvec: bool = True,
        rerank_scale_eigvec: bool = True,
        rerank_eigvec_layernorm: bool = False,
        rerank_num_sinkhorn_iterations: int = 100,
        rerank_output_mlp_ratio: float = 1.0,
        rerank_mutual_corr: bool = False,
        rerank_use_sc_score: bool = False,
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
            rerank_indices (tuple): Indices of feature pyramid to use for re-ranking (negative indices accepted).
            coarse_feat_embed_dim (int): Embedding dim for reranking features (using MLP), set None to disable.
            geometric_consistency_d_thresh (tuple): Distance threshold to use in geometric consistency adjacency matrix, per relay token level
            rerank_geotransformer_refinement (bool): Use geotransformer layer to refine features for re-ranking
            rerank_num_correspondences (tuple): Total number of local correspondences for geometric consistency re-ranking, per coarse level.
            rerank_sort_eigvec (bool): Sort eigenvector prior to MLP
            rerank_scale_eigvec (bool): Scale eigenvector to range [0, 1] prior to MLP
            rerank_eigvec_layernorm (bool): Instead apply layernorm to the eigvec priot to the MLP
            rerank_num_sinkhorn_iterations (int): Number of sinkhorn iterations. Set to 0 to disable OT.
            rerank_output_mlp_ratio (float): MLP ratio for rerank output (for hidden layer)
            rerank_mutual_corr (bool): Mutual correspondence filtering
            rerank_use_sc_score (bool): Add spatial consistency score as an additional feature to the MLP

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
        self.rerank_mode = rerank_mode
        self.rerank_indices = rerank_indices
        self.rerank_feat_embed_dim = rerank_feat_embed_dim
        self.geometric_consistency_d_thresh = geometric_consistency_d_thresh
        self.rerank_geotransformer_refinement = rerank_geotransformer_refinement
        self.rerank_num_correspondences = rerank_num_correspondences
        self.rerank_sort_eigvec = rerank_sort_eigvec
        self.rerank_scale_eigvec = rerank_scale_eigvec
        self.rerank_eigvec_layernorm = rerank_eigvec_layernorm
        self.rerank_num_sinkhorn_iterations = rerank_num_sinkhorn_iterations
        self.rerank_output_mlp_ratio = rerank_output_mlp_ratio
        self.rerank_mutual_corr = rerank_mutual_corr
        self.rerank_use_sc_score = rerank_use_sc_score
        if self.rerank_scale_eigvec and self.rerank_eigvec_layernorm:
            raise ValueError('Redundant choice of parameters, select one or the other')
        assert (
            len(self.rerank_indices)
            == len(self.rerank_feat_embed_dim)
            == len(self.geometric_consistency_d_thresh)
            == len(self.rerank_num_correspondences)
        ), 'Must have an entry for each re-ranking feat index'

        self.compute_rerank_depth_from_idx()
        self.get_rerank_input_dim()

        if self.rerank_mode == 'local_hierarchical_gc':
            if self.rerank_geotransformer_refinement:  # Use same MLP as geotrans
                self.rerank_coarse_feat_decoder = self.coarse_feat_decoder
            else:  # If not, create new MLP
                self.rerank_coarse_feat_decoder = nn.ModuleList()
                for ii, rerank_feat_input_dim in enumerate(self.rerank_feat_input_dim):
                    self.rerank_coarse_feat_decoder.append(
                        MLP(
                            rerank_feat_input_dim,
                            int(rerank_feat_input_dim * self.mlp_ratio),
                            self.rerank_feat_embed_dim[ii],
                        ) if self.rerank_feat_embed_dim is not None else nn.Identity()
                    )

            self.rerank_optimal_transport = None
            if self.rerank_num_sinkhorn_iterations > 0:
                self.rerank_optimal_transport = LearnableLogOptimalTransport(
                    self.rerank_num_sinkhorn_iterations
                )

            self.output_dim = (sum(self.rerank_num_correspondences)
                               + int(self.rerank_use_sc_score) * len(self.rerank_indices))
            self.output_mlp = MLP(self.output_dim, int(self.output_dim * self.rerank_output_mlp_ratio), 1)
            self.sigmoid = nn.Sigmoid()
            if self.rerank_eigvec_layernorm:
                self.eigvec_layernorms = nn.ModuleList(
                    [nn.LayerNorm(dim) for dim in self.rerank_num_correspondences]
                )
        elif self.rerank_mode == 'sgv':
            pass
        else:
            raise NotImplementedError

    def compute_rerank_depth_from_idx(self):
        """
        Determines the octree depth of reranking features based on the
        input octree depth and HOTFormerLoc parameters.
        """
        num_downsamples = (self.hotformerloc_global.backbone.backbone.stem_down
                           if self.hotformerloc_global.backbone.backbone.downsample_input_embeddings
                           else 0)
        num_stages = self.hotformerloc_global.backbone.backbone.num_stages
        depth_start = self.octree_depth - num_downsamples
        self.depth_rerank = []
        for rerank_idx in self.rerank_indices:
            if rerank_idx >= 0:
                depth_rerank = depth_start - rerank_idx
            else:  # neg idx
                depth_rerank = depth_start - num_stages - rerank_idx
            self.depth_rerank.append(depth_rerank)

    def get_rerank_input_dim(self):
        """
        Determing reranking feature input dimensions based on HOTFormerLoc
        parameters.
        """
        channels = list(self.hotformerloc_global.backbone.backbone.channels)
        num_octf_levels = self.hotformerloc_global.backbone.backbone.num_octf_levels
        num_pyramid_levels = self.hotformerloc_global.backbone.backbone.num_pyramid_levels
        if len(channels[num_octf_levels:]) == 1:
            channels[num_octf_levels:] = channels[num_octf_levels:] * num_pyramid_levels
        self.rerank_feat_input_dim = []
        for rerank_idx in self.rerank_indices:
            self.rerank_feat_input_dim.append(channels[rerank_idx])

    def local_hierarchical_gc_rerank(
        self,
        model_out: dict,
        hard_triplets: Tuple,
        shift_and_scale: Tensor,
        batch: dict,
        **kwargs,
    ):
        """
        Perform re-ranking of query and N candidates. Note that only 'local'
        keys are needed from HOTFormerLoc outputs.

        Args:
            model_out (dict): HOTFormerLoc output for entire (inference) batch. Assums first batch item is query, and all others are candidates.
            hard_triplets (tuple): Tuple of triplets of form (anc_indices, pos_indices, neg_indices)
            shift_and_scale (Tensor): (B, 4) tensor containing normalization parameters
            batch (dict): Batch containing `octree` and `points` objects

        Returns:
            rerank_dict (dict)
            targets (Tensor)
        """
        tic_start = time.perf_counter()
        time_dict = {}
        octree: Octree = batch['octree']
        points: Points = batch['points']
        anc_indices, pos_indices, neg_indices = hard_triplets

        # Process each pyramid level
        leading_eigvec_list = []
        sc_scores_list = []
        for rerank_ii, depth_j in enumerate(self.depth_rerank):
            # Get local points and features
            local_feats_depth_j = model_out['local'][depth_j]
            local_points_depth_j = get_octant_centroids_from_points(points, depth_j, self.quantizer)
            if shift_and_scale is not None:
                local_points_depth_j = Normalize.batch_unnormalize_concat(
                    local_points_depth_j, shift_and_scale, octree, depth_j
                )
            # Embed feats
            local_feats_depth_j = self.rerank_coarse_feat_decoder[rerank_ii](local_feats_depth_j)
            # Pad batched tensors and compute padding mask
            local_points_depth_j_padded = split_and_pad_data(
                octree, local_points_depth_j, depth_j, fill_value=self.point_padding,
            )
            local_feats_depth_j_padded, local_mask_depth_j_padded = split_and_pad_data(
                octree, local_feats_depth_j, depth_j, fill_value=0.0, return_mask=True
            )

            # Separate into anc and nn sets
            anc_points_depth_j_padded = local_points_depth_j_padded[anc_indices]
            pos_points_depth_j_padded = local_points_depth_j_padded[pos_indices]
            neg_points_depth_j_padded = local_points_depth_j_padded[neg_indices]
            anc_feats_depth_j_padded = local_feats_depth_j_padded[anc_indices]
            pos_feats_depth_j_padded = local_feats_depth_j_padded[pos_indices]
            neg_feats_depth_j_padded = local_feats_depth_j_padded[neg_indices]
            anc_mask_depth_j_padded = local_mask_depth_j_padded[anc_indices]
            pos_mask_depth_j_padded = local_mask_depth_j_padded[pos_indices]
            neg_mask_depth_j_padded = local_mask_depth_j_padded[neg_indices]
            B, N, C = anc_feats_depth_j_padded.shape  # recompute B here to get correct batch size
            NN = 2  # Only 2 nearest-neighbours during training (pos and neg)

            anc_points_depth_j_padded = anc_points_depth_j_padded[:, None, ...].expand(-1, NN, -1, -1)
            nn_points_depth_j_padded = torch.stack(
                (pos_points_depth_j_padded, neg_points_depth_j_padded), dim=1
            )
            anc_feats_depth_j_padded = anc_feats_depth_j_padded[:, None, ...].expand(-1, NN, -1, -1)
            nn_feats_depth_j_padded = torch.stack(
                (pos_feats_depth_j_padded, neg_feats_depth_j_padded), dim=1
            )
            anc_mask_depth_j_padded = anc_mask_depth_j_padded[:, None, ...].expand(-1, NN, -1)
            nn_mask_depth_j_padded = torch.stack(
                (pos_mask_depth_j_padded, neg_mask_depth_j_padded), dim=1
            )
 
            # Collapse nearest-neighbours into batch dimension (and reshape afterwards)
            anc_points_depth_j_padded = anc_points_depth_j_padded.reshape(B*NN, N, 3)
            nn_points_depth_j_padded = nn_points_depth_j_padded.reshape(B*NN, N, 3)
            anc_feats_depth_j_padded = anc_feats_depth_j_padded.reshape(B*NN, N, C)
            nn_feats_depth_j_padded = nn_feats_depth_j_padded.reshape(B*NN, N, C)
            anc_mask_depth_j_padded = anc_mask_depth_j_padded.reshape(B*NN, N)
            nn_mask_depth_j_padded = nn_mask_depth_j_padded.reshape(B*NN, N)
            
            # Process feats with geotrans
            if self.rerank_geotransformer_refinement:
                # NOTE: Memory usage is immense with large triplet BS and if significant padding is needed.
                #       Culprit seems to be geometric structure embedding layer
                assert self.coarse_feat_refiner is not None, 'Geotransformer must be enabled'
                tic = time.perf_counter()
                if self.grad_checkpoint and self.training:
                    anc_feats_depth_j_padded, nn_feats_depth_j_padded = checkpoint(
                        self.coarse_feat_refiner[rerank_ii],
                        anc_points_depth_j_padded,
                        nn_points_depth_j_padded,
                        anc_feats_depth_j_padded,
                        nn_feats_depth_j_padded,
                        anc_mask_depth_j_padded,
                        nn_mask_depth_j_padded,
                        use_reentrant=False,
                    )
                else:
                    anc_feats_depth_j_padded, nn_feats_depth_j_padded = self.coarse_feat_refiner[rerank_ii](
                        anc_points_depth_j_padded,
                        nn_points_depth_j_padded,
                        anc_feats_depth_j_padded,
                        nn_feats_depth_j_padded,
                        anc_mask_depth_j_padded,
                        nn_mask_depth_j_padded,
                    )
                time_dict[f'geotrans forward {rerank_ii}'] = time.perf_counter() - tic

            # Sinkhorn matching
            tic = time.perf_counter()
            if self.rerank_optimal_transport is not None:
                matching_scores_depth_j = torch.einsum('bnd,bmd->bnm', anc_feats_depth_j_padded, nn_feats_depth_j_padded)  # (B*NN, N, N)
                matching_scores_depth_j = matching_scores_depth_j / C ** 0.5
                if self.grad_checkpoint and self.training:
                    matching_scores_depth_j = checkpoint(
                        self.rerank_optimal_transport,
                        matching_scores_depth_j,
                        torch.logical_not(anc_mask_depth_j_padded),
                        torch.logical_not(nn_mask_depth_j_padded),
                        use_reentrant=False,
                    )
                else:
                    matching_scores_depth_j = self.rerank_optimal_transport(
                        matching_scores_depth_j,
                        torch.logical_not(anc_mask_depth_j_padded),
                        torch.logical_not(nn_mask_depth_j_padded),
                    )
                # Discard dustbin (and convert back from log-space)
                matching_scores_depth_j = torch.exp(matching_scores_depth_j[:, :-1, :-1])
                time_dict[f'optimal transport {rerank_ii}'] = time.perf_counter() - tic
            else:
                # Compute NN with cosine similarity
                anc_feats_depth_j_padded = torch.nn.functional.normalize(anc_feats_depth_j_padded, p=2.0, dim=-1)
                nn_feats_depth_j_padded = torch.nn.functional.normalize(nn_feats_depth_j_padded, p=2.0, dim=-1)
                matching_scores_depth_j = torch.einsum('bnd,bmd->bnm', anc_feats_depth_j_padded, nn_feats_depth_j_padded)  # (NN, N, N)
                mask = torch.logical_or(anc_mask_depth_j_padded[..., None], nn_mask_depth_j_padded[..., None, :])
                matching_scores_depth_j.masked_fill_(mask, -1e10)
                time_dict[f'feat matching {rerank_ii}'] = time.perf_counter() - tic
            
            k_corr = min(N, self.rerank_num_correspondences[rerank_ii])
            if self.rerank_mutual_corr:
                # Only consider mutual correspondences
                tic = time.perf_counter()
                k_mutual = 3
                matching_scores_depth_j = mutual_topk_correspondences(
                    matching_scores_depth_j, k_mutual, k_corr
                )
                time_dict[f'mutual nn {rerank_ii}'] = time.perf_counter() - tic
            # Consider NN from anc side
            anc_max_scores_depth_j, anc_max_indices_depth_j = matching_scores_depth_j.max(dim=2)  # (B*NN, N)
            _, nn_corr_indices_depth_j = anc_max_scores_depth_j.topk(
                k=k_corr, dim=1
            )
            anc_corr_indices_depth_j = anc_max_indices_depth_j.gather(dim=1, index=nn_corr_indices_depth_j)

            # Gather correspondences
            anc_corr_points_depth_j = anc_points_depth_j_padded.gather(dim=1, index=nn_corr_indices_depth_j[..., None].expand(-1, -1, 3))
            nn_corr_points_depth_j = nn_points_depth_j_padded.gather(dim=1, index=anc_corr_indices_depth_j[..., None].expand(-1, -1, 3))
            
            # Pad if needed
            if k_corr < self.rerank_num_correspondences[rerank_ii]:
                k_diff = self.rerank_num_correspondences[rerank_ii] - k_corr
                anc_corr_points_depth_j = F.pad(anc_corr_points_depth_j, (0, 0, 0, k_diff), value=self.point_padding)
                nn_corr_points_depth_j = F.pad(nn_corr_points_depth_j, (0, 0, 0, k_diff), value=self.point_padding)
            
            # # TODO: Add logic to re-compute if one batch elem has < num_corr (re-compute with topk=2|3 instead of max, allow violating one-to-one)
            # # NOTE: Not doing this for now, as it is a rare occurence, and masking padded points handles it gracefully enough.
            # anc_valid_point_counts = torch.logical_not(anc_mask_depth_j_padded).sum(dim=-1, keepdim=True) 
            # nn_valid_point_counts = torch.logical_not(nn_mask_depth_j_padded).sum(dim=-1, keepdim=True) 
            # anc_invalid_corr_mask = torch.any(anc_corr_indices_depth_j >= nn_valid_point_counts, dim=-1)
            # nn_invalid_corr_mask = torch.any(nn_corr_indices_depth_j >= anc_valid_point_counts, dim=-1)
            # invalid_corr_mask = torch.logical_or(anc_invalid_corr_mask, nn_invalid_corr_mask)
            # for invalid_batch_idx in invalid_corr_mask.nonzero():
                
            # Return points to original shape
            anc_corr_points_depth_j = anc_corr_points_depth_j.reshape(B, NN, k_corr, 3)
            nn_corr_points_depth_j = nn_corr_points_depth_j.reshape(B, NN, k_corr, 3)

            ### CLASSIFIER ###
            # Mask out any invalid correspondences (i.e. padding points)
            anc_sgv_mask_depth_j = (anc_corr_points_depth_j == self.point_padding).all(dim=-1)  # (B, NN, K)
            nn_sgv_mask_depth_j = (nn_corr_points_depth_j == self.point_padding).all(dim=-1)
            sgv_mask_depth_j = torch.logical_not(
                torch.logical_or(anc_sgv_mask_depth_j[..., None], nn_sgv_mask_depth_j[..., None, :])
            )
            
            # Compute spectral geometric consistency
            tic = time.perf_counter()
            leading_eigvec, spatial_consistency_score = batched_sgv_parallel(
                anc_corr_points_depth_j,
                nn_corr_points_depth_j,
                d_thresh=self.geometric_consistency_d_thresh[rerank_ii],
                return_spatial_consistency=True,
                mask=sgv_mask_depth_j,
            )
            time_dict[f'sgv {rerank_ii}'] = time.perf_counter() - tic
            if self.rerank_sort_eigvec:
                leading_eigvec, sort_indices = torch.sort(leading_eigvec, dim=-1, descending=True)

            if self.rerank_scale_eigvec:
                leading_eigvec = min_max_normalize(leading_eigvec)
            elif self.rerank_eigvec_layernorm:
                leading_eigvec = self.eigvec_layernorms[rerank_ii](leading_eigvec)
            if self.rerank_use_sc_score:
                # Append spatial consistency score as an additional feature
                leading_eigvec = torch.concat((leading_eigvec, spatial_consistency_score), dim=-1)
            leading_eigvec_list.append(leading_eigvec)
            sc_scores_list.append(spatial_consistency_score)

        # Concat eigvecs and pass through MLP + sigmoid to get scores
        rerank_features = torch.concat(leading_eigvec_list, dim=-1)
        rerank_scores = self.output_mlp(rerank_features)
        rerank_scores = self.sigmoid(rerank_scores)

        # Create target labels
        targets = torch.zeros_like(rerank_scores)  # [B, 2, 1]
        targets[:, 0] = 1  # set label of positives to 1

        toc = time.perf_counter()
        time_dict['TOTAL'] = toc - tic_start
        self.log_time_dict(time_dict, initial_str='Re-ranking:  ')
        out_dict = {
            'scores': rerank_scores, 'eigvec_list': rerank_features,
            'sc_scores': sc_scores_list
        }
        return out_dict, targets

    def local_hierarchical_gc_rerank_inference(
        self,
        model_out: dict,
        shift_and_scale: Tensor,
        batch: dict,
        **kwargs,
    ):
        """
        Perform re-ranking of query and N candidates. Note that only 'local'
        keys are needed from HOTFormerLoc outputs.

        Args:
            model_out (dict): HOTFormerLoc output for entire (inference) batch. Assums first batch item is query, and all others are candidates.
            shift_and_scale (Tensor): (B, 4) tensor containing normalization parameters
            batch (dict): Batch containing `octree` and `points` objects

        Returns:
            rerank_dict (dict)
        """
        tic_start = time.perf_counter()
        time_dict = {}
        octree: Octree = batch['octree']
        points: Points = batch['points']
        anc_idx = 0
        NN = octree.batch_size - 1  # first elem is query

        # Process each pyramid level
        leading_eigvec_list = []
        sc_scores_list = []
        for rerank_ii, depth_j in enumerate(self.depth_rerank):
            # Get local points and features
            if isinstance(model_out['local'], dict):  # standard HOTFloc output
                local_feats_depth_j = model_out['local'][depth_j]
            elif isinstance(model_out['local'], list):  # inference output, nested list of local feats from query and NNs
                local_feats_list_depth_j = [feat_dict[depth_j] for feat_dict in model_out['local']]
                local_feats_depth_j = torch.concat(local_feats_list_depth_j, dim=0)
                assert local_feats_depth_j.size(0) == octree.batch_nnum_nempty[depth_j].sum(), 'Octree does not match local feats'
            local_points_depth_j = get_octant_centroids_from_points(points, depth_j, self.quantizer)
            if shift_and_scale is not None:
                local_points_depth_j = Normalize.batch_unnormalize_concat(
                    local_points_depth_j, shift_and_scale, octree, depth_j
                )
            # Embed feats
            local_feats_depth_j = self.rerank_coarse_feat_decoder[rerank_ii](local_feats_depth_j)
            # Pad batched tensors and compute padding mask
            local_points_depth_j_padded = split_and_pad_data(
                octree, local_points_depth_j, depth_j, fill_value=self.point_padding,
            )
            local_feats_depth_j_padded, local_mask_depth_j_padded = split_and_pad_data(
                octree, local_feats_depth_j, depth_j, fill_value=0.0, return_mask=True
            )

            # Separate into anc and nn sets 
            anc_points_depth_j_padded = local_points_depth_j_padded[None, anc_idx].expand(NN, -1, -1)  # (NN, N, 3)
            nn_points_depth_j_padded = local_points_depth_j_padded[anc_idx+1:]  # (NN, N, 3)
            anc_feats_depth_j_padded = local_feats_depth_j_padded[None, anc_idx].expand(NN, -1, -1)  # (NN, N, C)
            nn_feats_depth_j_padded = local_feats_depth_j_padded[anc_idx+1:]  # (NN, N, C)
            anc_mask_depth_j_padded = local_mask_depth_j_padded[None, anc_idx].expand(NN, -1)  # (NN, N)
            nn_mask_depth_j_padded = local_mask_depth_j_padded[anc_idx+1:]  # (NN, N)
            _, N, C = anc_feats_depth_j_padded.shape  

            # Process feats with geotrans
            if self.rerank_geotransformer_refinement:
                # NOTE: Memory usage is immense with large triplet BS and if significant padding is needed.
                #       Culprit seems to be geometric structure embedding layer
                if self.coarse_feat_refiner[rerank_ii] is not None:
                    tic = time.perf_counter()
                    if self.grad_checkpoint and self.training:
                        anc_feats_depth_j_padded, nn_feats_depth_j_padded = checkpoint(
                            self.coarse_feat_refiner[rerank_ii],
                            anc_points_depth_j_padded,
                            nn_points_depth_j_padded,
                            anc_feats_depth_j_padded,
                            nn_feats_depth_j_padded,
                            anc_mask_depth_j_padded,
                            nn_mask_depth_j_padded,
                            use_reentrant=False,
                        )
                    else:
                        anc_feats_depth_j_padded, nn_feats_depth_j_padded = self.coarse_feat_refiner[rerank_ii](
                            anc_points_depth_j_padded,
                            nn_points_depth_j_padded,
                            anc_feats_depth_j_padded,
                            nn_feats_depth_j_padded,
                            anc_mask_depth_j_padded,
                            nn_mask_depth_j_padded,
                        )
                    time_dict[f'geotrans forward {rerank_ii}'] = time.perf_counter() - tic

            # Sinkhorn matching
            if self.rerank_optimal_transport is not None:
                tic = time.perf_counter()
                matching_scores_depth_j = torch.einsum('bnd,bmd->bnm', anc_feats_depth_j_padded, nn_feats_depth_j_padded)  # (NN, N, N)
                matching_scores_depth_j = matching_scores_depth_j / C ** 0.5
                if self.grad_checkpoint and self.training:
                    matching_scores_depth_j = checkpoint(
                        self.rerank_optimal_transport,
                        matching_scores_depth_j,
                        torch.logical_not(anc_mask_depth_j_padded),
                        torch.logical_not(nn_mask_depth_j_padded),
                        use_reentrant=False,
                    )
                else:
                    matching_scores_depth_j = self.rerank_optimal_transport(
                        matching_scores_depth_j,
                        torch.logical_not(anc_mask_depth_j_padded),
                        torch.logical_not(nn_mask_depth_j_padded),
                    )
                # Discard dustbin (and convert back from log-space)
                matching_scores_depth_j = torch.exp(matching_scores_depth_j[:, :-1, :-1])
                time_dict[f'optimal transport {rerank_ii}'] = time.perf_counter() - tic
            else:
                # Compute NN with cosine similarity
                anc_feats_depth_j_padded = torch.nn.functional.normalize(anc_feats_depth_j_padded, p=2.0, dim=-1)
                nn_feats_depth_j_padded = torch.nn.functional.normalize(nn_feats_depth_j_padded, p=2.0, dim=-1)
                matching_scores_depth_j = torch.einsum('bnd,bmd->bnm', anc_feats_depth_j_padded, nn_feats_depth_j_padded)  # (NN, N, N)
                mask = torch.logical_or(anc_mask_depth_j_padded[..., None], nn_mask_depth_j_padded[..., None, :])
                matching_scores_depth_j.masked_fill_(mask, -1e10)
            
            k_corr = min(N, self.rerank_num_correspondences[rerank_ii])
            if self.rerank_mutual_corr:
                # Only consider mutual correspondences
                tic = time.perf_counter()
                k_mutual = 3
                matching_scores_depth_j = mutual_topk_correspondences(
                    matching_scores_depth_j, k_mutual, k_corr
                )
                time_dict[f'mutual nn {rerank_ii}'] = time.perf_counter() - tic
            # Consider NN from anc side
            anc_max_scores_depth_j, anc_max_indices_depth_j = matching_scores_depth_j.max(dim=2)  # (NN, N)
            _, nn_corr_indices_depth_j = anc_max_scores_depth_j.topk(
                k=k_corr, dim=1
            )
            anc_corr_indices_depth_j = anc_max_indices_depth_j.gather(dim=1, index=nn_corr_indices_depth_j)

            # Gather correspondences
            anc_corr_points_depth_j = anc_points_depth_j_padded.gather(dim=1, index=nn_corr_indices_depth_j[..., None].expand(-1, -1, 3))
            nn_corr_points_depth_j = nn_points_depth_j_padded.gather(dim=1, index=anc_corr_indices_depth_j[..., None].expand(-1, -1, 3))
            
            # Pad if needed
            if k_corr < self.rerank_num_correspondences[rerank_ii]:
                k_diff = self.rerank_num_correspondences[rerank_ii] - k_corr
                anc_corr_points_depth_j = F.pad(anc_corr_points_depth_j, (0, 0, 0, k_diff), value=self.point_padding)
                nn_corr_points_depth_j = F.pad(nn_corr_points_depth_j, (0, 0, 0, k_diff), value=self.point_padding)
            
            # # TODO: Add logic to re-compute if one batch elem has < num_corr (re-compute with topk=2|3 instead of max, allow violating one-to-one)
            # # NOTE: Not doing this for now, as it is a rare occurence, and masking padded points handles it gracefully enough.
            # anc_valid_point_counts = torch.logical_not(anc_mask_depth_j_padded).sum(dim=-1, keepdim=True) 
            # nn_valid_point_counts = torch.logical_not(nn_mask_depth_j_padded).sum(dim=-1, keepdim=True) 
            # anc_invalid_corr_mask = torch.any(anc_corr_indices_depth_j >= nn_valid_point_counts, dim=-1)
            # nn_invalid_corr_mask = torch.any(nn_corr_indices_depth_j >= anc_valid_point_counts, dim=-1)
            # invalid_corr_mask = torch.logical_or(anc_invalid_corr_mask, nn_invalid_corr_mask)
            # for invalid_batch_idx in invalid_corr_mask.nonzero():
                
            ### CLASSIFIER ###
            # Mask out any invalid correspondences (i.e. padding points)
            anc_sgv_mask_depth_j = (anc_corr_points_depth_j == self.point_padding).all(dim=-1)  # (NN, K)
            nn_sgv_mask_depth_j = (nn_corr_points_depth_j == self.point_padding).all(dim=-1)
            sgv_mask_depth_j = torch.logical_not(
                torch.logical_or(anc_sgv_mask_depth_j[..., None], nn_sgv_mask_depth_j[..., None, :])
            )
            
            # Compute spectral geometric consistency
            tic = time.perf_counter()
            leading_eigvec, spatial_consistency_score = batched_sgv_parallel(
                anc_corr_points_depth_j[None, ...],
                nn_corr_points_depth_j[None, ...],
                d_thresh=self.geometric_consistency_d_thresh[rerank_ii],
                return_spatial_consistency=True,
                mask=sgv_mask_depth_j,
            )
            time_dict[f'sgv {rerank_ii}'] = time.perf_counter() - tic
            if self.rerank_sort_eigvec:
                leading_eigvec, sort_indices = torch.sort(leading_eigvec, dim=-1, descending=True)

            if self.rerank_scale_eigvec:
                leading_eigvec = min_max_normalize(leading_eigvec)
            elif self.rerank_eigvec_layernorm:
                leading_eigvec = self.eigvec_layernorms[rerank_ii](leading_eigvec)
            if self.rerank_use_sc_score:
                # Append spatial consistency score as an additional feature
                leading_eigvec = torch.concat((leading_eigvec, spatial_consistency_score), dim=-1)
            leading_eigvec_list.append(leading_eigvec)
            sc_scores_list.append(spatial_consistency_score)

        # Concat eigvecs and pass through MLP + sigmoid to get scores
        rerank_features = torch.concat(leading_eigvec_list, dim=-1)
        rerank_scores = self.output_mlp(rerank_features)
        rerank_scores = self.sigmoid(rerank_scores)  # (1, NN, 1)

        toc = time.perf_counter()
        time_dict['TOTAL'] = toc - tic_start
        self.log_time_dict(time_dict, initial_str='Re-ranking:  ')
        return {'scores': rerank_scores, 'eigvec_list': rerank_features,
                'sc_scores': sc_scores_list}

    def rerank(self, *args, **kwargs):
        if self.rerank_mode is None:
            return self.hotformerloc_global.rerank(*args, **kwargs)
        elif self.rerank_mode == 'local_hierarchical_gc':
            return self.local_hierarchical_gc_rerank(*args, **kwargs)
        elif self.rerank_mode == 'sgv':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def rerank_inference(self, *args, **kwargs):
        if self.rerank_mode is None:
            return self.hotformerloc_global.rerank_inference(*args, **kwargs)
        elif self.rerank_mode == 'local_hierarchical_gc':
            return self.local_hierarchical_gc_rerank_inference(*args, **kwargs)
        elif self.rerank_mode == 'sgv':
            return self.sgv_rerank_inference(*args, **kwargs)
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
