"""
Relay token re-ranker with geometric consistency.

Ethan Griffiths (Data61, Pullenvale)
"""
from typing import Tuple, Optional
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ocnn.octree import Points

from dataset.augmentation import Normalize
from dataset.coordinate_utils import CoordinateSystem
from models.octree import OctreeT, get_octant_centroids_from_points, split_and_pad_data, unpad_and_concat_data
from models.layers.octformer_layers import MLP
from models.relay_token_utils import concat_and_pad_rt, unpad_and_split_rt
from models.reranking_utils import batched_sgv_parallel

class RelayTokenGeometricConsistencyReranker(torch.nn.Module):
    """
    Relay token re-ranker with geometric consistency.

    Args:
        rerank_rt_indices (tuple): Indices of relay token levels to use for re-ranking
        attn_topk (tuple): Number of top-k relay tokens to select based on CLS attn maps, per level
        geometric_consistency_d_thresh (tuple): Distance threshold to use in geometric consistency adjacency matrix, per relay token level
        rt_dim (int): Dimension of relay tokens 
        sort_eigvec (bool): Sort eigenvectors prior to processing in MLP
        use_attn_vals (bool): Also use relay token attn values as a feature in the MLP classifier
    """

    def __init__(
        self,
        rerank_rt_indices: Tuple[int],
        attn_topk: Tuple[int],
        geometric_consistency_d_thresh: Tuple[float],  # metres
        rt_dim: int,
        sort_eigvec: bool = True,
        use_attn_vals: bool = False,
    ):
        super().__init__()
        self.rerank_rt_indices = rerank_rt_indices
        self.attn_topk = attn_topk
        if self.attn_topk is None:
            raise ValueError('`attn_topk` currently required to ensure consistent number of relay tokens')
        self.geometric_consistency_d_thresh = geometric_consistency_d_thresh
        if len(attn_topk) != len(rerank_rt_indices) != len(geometric_consistency_d_thresh):
            raise ValueError('`attn_topk` must have same num elems as `rerank_rt_indices` and `geometric_consistency_d_thresh`')
        self.rt_dim = rt_dim
        self.sort_eigvec = sort_eigvec
        self.use_attn_vals = use_attn_vals
        
        self.input_linear = nn.Linear(self.rt_dim, self.rt_dim)
        self.output_dim = sum(self.attn_topk) + (2 * int(use_attn_vals) * sum(self.attn_topk))  # concat attn vals to feat dim of MLP
        self.output_mlp = MLP(self.output_dim, self.output_dim, 1)  # TODO: different hidden size?
        self.sigmoid = nn.Sigmoid()

    def forward(self, model_out: dict, hard_triplets: Tuple, shift_and_scale: Tensor, *args, **kwargs):
        """
        Perform re-ranking of query and N candidates. Note that only 'octree',
        'rt', and 'rt_final_cls_attn_vals' keys are needed from HOTFormerLoc outputs.

        Args:
            model_out (dict): HOTFormerLoc output for entire (training) batch
            hard_triplets (tuple): Tuple of triplets of form (anc_indices, pos_indices, neg_indices)
            shift_and_scale (Tensor): (B, 4) tensor containing normalization parameters

        Returns:
            rerank_scores (Tensor)
            targets (Tensor)
        """
        tic = time.perf_counter()
        # Collect batch relay tokens
        octree: OctreeT = model_out['octree']
        anc_indices, pos_indices, neg_indices = hard_triplets
        rt_cls_attn = model_out['rt_final_cls_attn_vals']
        if self.attn_topk is not None:
            # Split back to each depth
            rt_cls_attn_per_depth = unpad_and_split_rt(rt_cls_attn[..., None], octree)

        # Process each RT level
        leading_eigvec_list = []
        anc_pos_attn_scores_list = []
        anc_neg_attn_scores_list = []
        for ii, rt_idx in enumerate(self.rerank_rt_indices):
            depth = octree.pyramid_depths[rt_idx]
            rt_depth_j = concat_and_pad_rt(model_out['rt'], octree, [depth])
            B, N, C = rt_depth_j.shape
            rt_centroids_depth_j = concat_and_pad_rt(octree.window_stats, octree, [depth])[..., :3]
            rt_mask = octree.rt_mask[depth][..., 0] == 0.  # Mask of non-padding RTs
            # Unnormalize RT centroids
            if shift_and_scale is not None:
                rt_centroids_depth_j = Normalize.batch_unnormalize(
                    rt_centroids_depth_j, shift_and_scale, mask=rt_mask
                )
            # Filter top-k attn vals for each depth
            if self.attn_topk is not None:
                rt_cls_attn_depth_j = concat_and_pad_rt(rt_cls_attn_per_depth, octree, [depth])
                if N < self.attn_topk[ii]:  # zero-pad
                    padding_size = self.attn_topk[ii] - N
                    rt_cls_attn_depth_j = F.pad(rt_cls_attn_depth_j, (0,0,0,padding_size))
                    rt_centroids_depth_j = F.pad(rt_centroids_depth_j, (0,0,0,padding_size))
                    rt_depth_j = F.pad(rt_depth_j, (0,0,0,padding_size))
                attn_topk_scores, attn_topk_indices = torch.topk(rt_cls_attn_depth_j, k=self.attn_topk[ii], dim=1, sorted=False)
                attn_topk_scores.squeeze_(-1)
                attn_topk_indices = torch.sort(attn_topk_indices, dim=1).values  # Sort indices to retain original ordering (ensures padding remains at end)
                rt_centroids_depth_j = torch.gather(rt_centroids_depth_j, dim=1, index=attn_topk_indices.expand(-1, -1, 3))
                rt_depth_j = torch.gather(rt_depth_j, dim=1, index=attn_topk_indices.expand(-1, -1, C))
                # Separate attn scores into anc and nn sets
                if self.use_attn_vals:
                    anc_attn_scores = attn_topk_scores[anc_indices]
                    pos_attn_scores = attn_topk_scores[pos_indices]
                    neg_attn_scores = attn_topk_scores[neg_indices]

            # Separate RTs into anc and nn sets
            anc_rt_centroids = rt_centroids_depth_j[anc_indices]
            nn_rt_centroids = torch.stack(
                [rt_centroids_depth_j[pos_indices], rt_centroids_depth_j[neg_indices]],
                dim=1,
            )
            anc_rt = rt_depth_j[anc_indices]
            nn_rt = torch.stack(
                [rt_depth_j[pos_indices], rt_depth_j[neg_indices]],
                dim=1,
            )

            # Compute RT mask for anc and nn
            rt_sgv_mask = compute_rt_sgv_mask(
                anc_rt, anc_indices, pos_indices, neg_indices, octree, depth
            )

            # Apply linear layer to RTs
            anc_rt = self.input_linear(anc_rt)
            nn_rt = self.input_linear(nn_rt)
            
            # Compute spectral geometric consistency
            leading_eigvec, spatial_consistency_score = batched_sgv_parallel(
                anc_rt_centroids,
                nn_rt_centroids,
                anc_rt,
                nn_rt,
                d_thresh=self.geometric_consistency_d_thresh[ii],
                mask=rt_sgv_mask,
                return_spatial_consistency=True,
            )
            if self.sort_eigvec:
                leading_eigvec, sort_indices = torch.sort(leading_eigvec, dim=-1, descending=True)

            if self.use_attn_vals:
                # Also sort attn vals with same index to ensure they correspond to the same relay tokens
                if self.sort_eigvec:
                    anc_pos_attn_scores_temp = torch.concat(
                        [torch.gather(anc_attn_scores, dim=-1, index=sort_indices[:,0,:]),
                         torch.gather(pos_attn_scores, dim=-1, index=sort_indices[:,0,:])],
                        dim=-1,
                    )
                    anc_neg_attn_scores_temp = torch.concat(
                        [torch.gather(anc_attn_scores, dim=-1, index=sort_indices[:,1,:]),
                         torch.gather(neg_attn_scores, dim=-1, index=sort_indices[:,1,:])],
                        dim=-1,
                    )
                else:
                    anc_pos_attn_scores_temp = torch.concat(
                        [anc_attn_scores, pos_attn_scores], dim=-1
                    )
                    anc_neg_attn_scores_temp = torch.concat(
                        [anc_attn_scores, neg_attn_scores], dim=-1
                    )

                anc_pos_attn_scores_list.append(anc_pos_attn_scores_temp)
                anc_neg_attn_scores_list.append(anc_neg_attn_scores_temp)

            # TODO: Scale eigvec? (softmax? or just l2norm?)
            leading_eigvec_list.append(leading_eigvec)

        # Concat eigvecs and pass through MLP + sigmoid to get scores
        rerank_features = torch.concat(leading_eigvec_list, dim=-1)
        if self.use_attn_vals:
            # Add RT attn values to re-ranking features
            anc_pos_attn_scores = torch.concat(anc_pos_attn_scores_list, dim=-1)
            anc_neg_attn_scores = torch.concat(anc_neg_attn_scores_list, dim=-1)
            # TODO: Check order of attn scores. Should it be (anc_RT0, anc_RT1, pos_RT0, pos_RT1),
            #       or (anc_RT0, posRT0, anc_RT1, pos_RT1)?
            #       Currently is the latter, and order shouldn't matter for linear layers regardless
            batch_attn_scores = torch.stack([anc_pos_attn_scores, anc_neg_attn_scores], dim=1)
            rerank_features = torch.concat([rerank_features, batch_attn_scores], dim=-1)

        rerank_scores = self.output_mlp(rerank_features)
        rerank_scores = self.sigmoid(rerank_scores)

        # Create target labels
        targets = torch.zeros_like(rerank_scores)  # [B, 2, 1]
        targets[:, 0] = 1  # set label of positives to 1

        toc = time.perf_counter()
        log_str = f'Re-ranking time: {toc-tic:.4f}s'
        logging.debug(log_str)
        return rerank_scores, targets

    def rerank_inference(self, model_out: dict, shift_and_scale: Tensor, *args, **kwargs):
        """
        Perform re-ranking of query and N candidates. Note that only 'octree',
        'rt', and 'rt_final_cls_attn_vals' keys are needed from HOTFormerLoc outputs.

        Args:
            model_out (dict): HOTFormerLoc output for entire (inference) batch. Assums first batch item is query, and all others are candidates.
            shift_and_scale (Tensor): (B, 4) tensor containing normalization parameters

        Returns:
            rerank_scores (Tensor)
        """
        tic = time.perf_counter()
        # Collect batch relay tokens
        octree: OctreeT = model_out['octree']
        anc_idx = 0
        NN = octree.batch_size - 1  # first elem is query
        rt_cls_attn = model_out['rt_final_cls_attn_vals']
        if self.attn_topk is not None:
            # Split back to each depth
            rt_cls_attn_per_depth = unpad_and_split_rt(rt_cls_attn[..., None], octree)

        # Process each RT level
        leading_eigvec_list = []
        anc_nn_attn_scores_list = []
        for ii, rt_idx in enumerate(self.rerank_rt_indices):
            depth = octree.pyramid_depths[rt_idx]
            rt_depth_j = concat_and_pad_rt(model_out['rt'], octree, [depth])
            B, N, C = rt_depth_j.shape
            rt_centroids_depth_j = concat_and_pad_rt(octree.window_stats, octree, [depth])[..., :3]
            rt_mask = octree.rt_mask[depth][..., 0] == 0.  # Mask of non-padding RTs
            # Unnormalize RT centroids
            if shift_and_scale is not None:
                rt_centroids_depth_j = Normalize.batch_unnormalize(
                    rt_centroids_depth_j, shift_and_scale, mask=rt_mask
                )
            # Filter top-k attn vals for each depth
            if self.attn_topk is not None:
                rt_cls_attn_depth_j = concat_and_pad_rt(rt_cls_attn_per_depth, octree, [depth])
                if N < self.attn_topk[ii]:  # zero-pad
                    padding_size = self.attn_topk[ii] - N
                    rt_cls_attn_depth_j = F.pad(rt_cls_attn_depth_j, (0,0,0,padding_size))
                    rt_centroids_depth_j = F.pad(rt_centroids_depth_j, (0,0,0,padding_size))
                    rt_depth_j = F.pad(rt_depth_j, (0,0,0,padding_size))
                attn_topk_scores, attn_topk_indices = torch.topk(rt_cls_attn_depth_j, k=self.attn_topk[ii], dim=1, sorted=False)
                attn_topk_scores.squeeze_(-1)
                attn_topk_indices = torch.sort(attn_topk_indices, dim=1).values  # Sort indices to retain original ordering (ensures padding remains at end)
                rt_centroids_depth_j = torch.gather(rt_centroids_depth_j, dim=1, index=attn_topk_indices.expand(-1, -1, 3))
                rt_depth_j = torch.gather(rt_depth_j, dim=1, index=attn_topk_indices.expand(-1, -1, C))
                # Separate attn scores into anc and nn sets
                if self.use_attn_vals:
                    anc_attn_scores = attn_topk_scores[None, None, anc_idx].expand(-1, NN, -1)
                    nn_attn_scores = attn_topk_scores[None, anc_idx+1:]

            # Separate RTs into anc and nn sets (expand batch dim to 1 for compatibility)
            anc_rt_centroids = rt_centroids_depth_j[None, None, anc_idx]
            nn_rt_centroids = rt_centroids_depth_j[None, anc_idx+1:]
            anc_rt = rt_depth_j[None, None, anc_idx]
            nn_rt = rt_depth_j[None, anc_idx+1:]

            # Compute RT mask for anc and nn
            rt_sgv_mask = compute_rt_sgv_mask_inference(
                nn_rt, anc_idx, octree, depth
            )

            # Apply linear layer to RTs
            anc_rt = self.input_linear(anc_rt)
            nn_rt = self.input_linear(nn_rt)
            
            # Compute spectral geometric consistency
            leading_eigvec, spatial_consistency_score = batched_sgv_parallel(
                anc_rt_centroids,
                nn_rt_centroids,
                anc_rt,
                nn_rt,
                d_thresh=self.geometric_consistency_d_thresh[ii],
                mask=rt_sgv_mask,
                return_spatial_consistency=True,
            )
            if self.sort_eigvec:
                leading_eigvec, sort_indices = torch.sort(leading_eigvec, dim=-1, descending=True)

            if self.use_attn_vals:
                # Also sort attn vals with same index to ensure they correspond to the same relay tokens
                if self.sort_eigvec:
                    anc_nn_attn_scores_temp = torch.concat(
                        [torch.gather(anc_attn_scores, dim=-1, index=sort_indices),
                         torch.gather(nn_attn_scores, dim=-1, index=sort_indices)],
                        dim=-1,
                    )
                else:
                    anc_nn_attn_scores_temp = torch.concat(
                        [anc_attn_scores, nn_attn_scores], dim=-1
                    )

                anc_nn_attn_scores_list.append(anc_nn_attn_scores_temp)

            # TODO: Scale eigvec? (softmax? or just l2norm?)
            leading_eigvec_list.append(leading_eigvec)

        # Concat eigvecs and pass through MLP + sigmoid to get scores
        rerank_features = torch.concat(leading_eigvec_list, dim=-1)
        if self.use_attn_vals:
            # Add RT attn values to re-ranking features
            anc_nn_attn_scores = torch.concat(anc_nn_attn_scores_list, dim=-1)
            # TODO: Check order of attn scores. Should it be (anc_RT0, anc_RT1, pos_RT0, pos_RT1),
            #       or (anc_RT0, posRT0, anc_RT1, pos_RT1)?
            #       Currently is the latter, and order shouldn't matter for linear layers regardless
            rerank_features = torch.concat([rerank_features, anc_nn_attn_scores], dim=-1)

        rerank_scores = self.output_mlp(rerank_features)
        rerank_scores = self.sigmoid(rerank_scores)

        toc = time.perf_counter()
        log_str = f'Re-ranking time: {toc-tic:.4f}s'
        logging.debug(log_str)
        return rerank_scores


class RelayTokenLocalGeometricConsistencyReranker(torch.nn.Module):
    """
    Relay token re-ranker with geometric consistency on local features. As 
    opposed to RelayTokenGCReranker, this class only uses relay tokens to 
    select salient local features for geometric consistency re-ranking, which
    provides more stable centroids than using relay tokens alone.

    Args:
        rerank_rt_indices (tuple): Indices of relay token levels to use for re-ranking.
        geometric_consistency_d_thresh (tuple): Distance threshold to use in geometric
            consistency adjacency matrix, per relay token level.
        rt_dim (int): Dimension of relay tokens.
        local_dims (tuple): Dimension of local features, per relay token level
        quantizer (CoordinateSystem): Optional quantizer class, used to undo conversion to cylindrical coordinates
        num_correspondences (tuple): Total number of local correspondences for geometric
            consistency, per relay token level.
        min_correspondences_per_window (tuple): Minimum number of local correspondences per
            attention window to use for geoemetric consistency, per relay token level.
        sort_eigvec (bool): Sort eigenvectors prior to processing in MLP.
        mutual (bool): Sort RT correspondences by mutual NN score
        
    """

    def __init__(
        self,
        rerank_rt_indices: Tuple[int],
        # attn_topk: Tuple[int],
        # rt_topk_correspondences: Tuple[int],
        geometric_consistency_d_thresh: Tuple[float],  # metres
        rt_dim: int,
        local_dims: Tuple[int],
        quantizer: Optional[CoordinateSystem] = None,
        num_correspondences: Tuple[int] = (128, 64),
        min_correspondences_per_window: Tuple[int] = (16, 16),
        sort_eigvec: bool = True,
        # use_attn_vals: bool = False,
        mutual: bool = False,
    ):
        super().__init__()
        self.rerank_rt_indices = rerank_rt_indices
        # self.attn_topk = attn_topk
        # if self.attn_topk is None:
        #     raise ValueError('`attn_topk` currently required to ensure consistent number of relay tokens')
        self.geometric_consistency_d_thresh = geometric_consistency_d_thresh
        self.rt_dim = rt_dim
        self.local_dims = local_dims
        self.quantizer = quantizer
        self.num_correspondences = num_correspondences
        self.min_correspondences_per_window = min_correspondences_per_window
        if (
            len(rerank_rt_indices)
            != len(geometric_consistency_d_thresh)
            != len(local_dims)
            != len(num_correspondences)
            != len(min_correspondences_per_window)
        ):
            raise ValueError(
                'Ensure same num elems for `rerank_rt_indices`, `geometric_consistency_d_thresh`, '
                '`num_correspondences`, `min_correspondences_per_window`'
            )
        self.sort_eigvec = sort_eigvec
        self.mutual = mutual
        self.similarity_mask_val = -1e3
        
        self.input_rt_linear = nn.Linear(self.rt_dim, self.rt_dim)
        self.input_local_linears = nn.ModuleList(
            [nn.Linear(local_dim, local_dim) for local_dim in self.local_dims]
        )
        self.output_dim = sum(self.num_correspondences)
        self.output_mlp = MLP(self.output_dim, self.output_dim, 1)  # TODO: different hidden size?
        self.sigmoid = nn.Sigmoid()

    def forward(self, model_out: dict, hard_triplets: Tuple, shift_and_scale: Tensor, points: Points, *args, **kwargs):
        """
        Perform re-ranking of query and N candidates. Note that only 'octree',
        'rt', and 'rt_final_cls_attn_vals' keys are needed from HOTFormerLoc outputs.

        Args:
            model_out (dict): HOTFormerLoc output for entire (training) batch
            hard_triplets (tuple): Tuple of triplets of form (anc_indices, pos_indices, neg_indices)
            shift_and_scale (Tensor): (B, 4) tensor containing normalization parameters
            points (Points): Points object containing original point cloud

        Returns:
            rerank_scores (Tensor)
            targets (Tensor)
        """
        tic = time.perf_counter()
        octree: OctreeT = model_out['octree']
        anc_indices, pos_indices, neg_indices = hard_triplets

        # Process each pyramid level
        leading_eigvec_list = []
        for ii, rt_idx in enumerate(self.rerank_rt_indices):
            ####################################################################
            ### PRE-PROCESSING
            ####################################################################
            depth = octree.pyramid_depths[rt_idx]
            rt_depth_j = concat_and_pad_rt(model_out['rt'], octree, [depth])
            rt_mask = octree.rt_mask[depth][..., 0] == 0.0  # Mask of non-padding RTs
            # Apply linear layers + L2 normalize (for computing cosine similarity later)
            rt_depth_j = self.input_rt_linear(rt_depth_j)
            rt_depth_j = F.normalize(rt_depth_j, p=2, dim=-1)
            rt_depth_j.masked_fill_(torch.logical_not(rt_mask[..., None]), value=0)
            # Separate RTs into anc and nn sets
            anc_rt = rt_depth_j[anc_indices]
            nn_rt = torch.stack(
                [rt_depth_j[pos_indices], rt_depth_j[neg_indices]],
                dim=1,
            )
            # Compute RT mask for anc and nn
            rt_sgv_mask = compute_rt_sgv_mask(
                anc_rt, anc_indices, pos_indices, neg_indices, octree, depth
            )
            # Get local points and features
            local_feats_depth_j = model_out['local'][depth]
            local_points_depth_j = get_octant_centroids_from_points(points, depth, self.quantizer)
            if shift_and_scale is not None:
                local_points_depth_j = Normalize.batch_unnormalize_concat(
                    local_points_depth_j, shift_and_scale, octree, depth
                )
            # Apply linear layers + L2 normalize (for computing cosine similarity later)
            local_feats_depth_j = self.input_local_linears[ii](local_feats_depth_j)
            local_feats_depth_j = F.normalize(local_feats_depth_j, p=2, dim=-1)
            # Pad batched tensors and compute padding mask
            local_points_depth_j_padded = split_and_pad_data(
                octree, local_points_depth_j, depth, fill_value=0.0,
            )
            local_feats_depth_j_padded, local_mask_depth_j_padded = split_and_pad_data(
                octree, local_feats_depth_j, depth, fill_value=0.0, return_mask=True
            )
            B_orig, N, C =  local_feats_depth_j_padded.shape
            K = octree.patch_size
            pad_len = (K - (N % K)) % K  # computes amount of padding needed to reshape into (B, N//K, K, C)
            local_points_depth_j_padded_windows = F.pad(
                local_points_depth_j_padded, (0, 0, 0, pad_len), value=0.0
            ).view(B_orig, -1, K, 3)
            local_feats_depth_j_padded_windows = F.pad(
                local_feats_depth_j_padded, (0, 0, 0, pad_len), value=0.0
            ).view(B_orig, -1, K, C)
            local_mask_depth_j_padded_windows = F.pad(
                local_mask_depth_j_padded, (0, pad_len), value=True
            ).view(B_orig, -1, K)
            
            # Separate into anc and nn sets
            anc_points_depth_j_padded = local_points_depth_j_padded_windows[anc_indices][:, None, ...].expand(-1, 2, -1, -1, -1)
            B = anc_points_depth_j_padded.size(0)  # recompute B to get correct batch size
            nn_points_depth_j_padded = torch.stack(
                [local_points_depth_j_padded_windows[pos_indices], local_points_depth_j_padded_windows[neg_indices]],
                dim=1,
            )
            anc_feats_depth_j_padded = local_feats_depth_j_padded_windows[anc_indices][:, None, ...].expand(-1, 2, -1, -1, -1)
            nn_feats_depth_j_padded = torch.stack(
                [local_feats_depth_j_padded_windows[pos_indices], local_feats_depth_j_padded_windows[neg_indices]],
                dim=1,
            )
            anc_mask_depth_j_padded = local_mask_depth_j_padded_windows[anc_indices][:, None, ...].expand(-1, 2, -1, -1)
            nn_mask_depth_j_padded = torch.stack(
                [local_mask_depth_j_padded_windows[pos_indices], local_mask_depth_j_padded_windows[neg_indices]],
                dim=1,
            )
            
            ####################################################################
            ### COMPUTE CORRESPONDENCES
            ####################################################################
            anc_rt = anc_rt[:, None, ...].expand(-1, 2, -1, -1)  # match shape of nn_rt
            rt_similarity_matrix = torch.matmul(  # (*, N, C) x (*, C, M) -> (*, N, M)
                anc_rt, nn_rt.transpose(-1, -2)
            )
            rt_similarity_matrix.masked_fill_(
                torch.logical_not(rt_sgv_mask), self.similarity_mask_val
            )

            # Determine num relay tokens required to reach num correspondences
            min_num_rt = min(octree.batch_num_rt_no_padding[depth]).item()
            correspondences_per_window = self.min_correspondences_per_window[ii]
            if self.num_correspondences[ii] // correspondences_per_window > min_num_rt:
                # Need to collect more correspondences per window to reach target
                # NOTE: maybe better to do this per batch item, but would be slower
                correspondences_per_window = self.num_correspondences[ii] // min_num_rt
            num_rt_correspondences = self.num_correspondences[ii] // correspondences_per_window
            
            # TODO: First compute NN with .max for each anc RT, then run top-k on the scores
            #       Alternatively, can do mutual NN (see point_matching.py)
            if self.mutual:
                raise NotImplementedError
            else:
                # Just use top-k correspondences from anc side
                # NOTE: Currently this doesn't guarantee a one-to-one mapping,
                #       so the same RT window could be used multiple times.
                anc_rt_correspondence_scores, anc_rt_correspondence_indices = torch.max(rt_similarity_matrix, dim=-1)
                _, nn_rt_topk_correspondence_indices = torch.topk(
                    anc_rt_correspondence_scores, k=num_rt_correspondences, dim=-1
                )
                anc_rt_topk_correspondence_indices = torch.gather(
                    anc_rt_correspondence_indices, dim=-1, index=nn_rt_topk_correspondence_indices
                )

            # Select points and feats within top-k windows
            anc_topk_points_depth_j_padded = torch.gather(
                anc_points_depth_j_padded, dim=2, index=nn_rt_topk_correspondence_indices[..., None, None].expand(-1, -1, -1, K, 3)
            )
            anc_topk_feats_depth_j_padded = torch.gather(
                anc_feats_depth_j_padded, dim=2, index=nn_rt_topk_correspondence_indices[..., None, None].expand(-1, -1, -1, K, C)
            )
            anc_topk_mask_depth_j_padded = torch.gather(
                anc_mask_depth_j_padded, dim=2, index=nn_rt_topk_correspondence_indices[..., None].expand(-1, -1, -1, K)
            )
            nn_topk_points_depth_j_padded = torch.gather(
                nn_points_depth_j_padded, dim=2, index=anc_rt_topk_correspondence_indices[..., None, None].expand(-1, -1, -1, K, 3)
            )
            nn_topk_feats_depth_j_padded = torch.gather(
                nn_feats_depth_j_padded, dim=2, index=anc_rt_topk_correspondence_indices[..., None, None].expand(-1, -1, -1, K, C)
            )
            nn_topk_mask_depth_j_padded = torch.gather(
                nn_mask_depth_j_padded, dim=2, index=anc_rt_topk_correspondence_indices[..., None].expand(-1, -1, -1, K)
            )

            # # DEBUGGING VISUALISATIONS
            # from misc.point_clouds import plot_points, plot_registration_result                
            # plot_points(anc_points_depth_j_padded[-1,0].view(-1,3).cpu()) # all anc points
            # plot_points(nn_points_depth_j_padded[-1,0].view(-1,3).cpu())  # all pos points
            # plot_points(anc_topk_points_depth_j_padded[-1,0].view(-1,3).cpu())  # top-k anc windows
            # plot_points(nn_topk_points_depth_j_padded[-1,0].view(-1,3).cpu())  # top-k pos windows

            # octree.batch_nnum_nempty[depth]
            # octree.batch_id(depth, True)
            # octree.batch_num_windows[depth]
            # octree.batch_num_rt_no_padding[depth]
            # # local_feat_depth_j.split(octree.batch_num_windows[depth].tolist())
            # octree.patch_mask[depth]

            # Compute pair-wise cosine similarity within window correspondences
            point_similarity_matrix = torch.matmul(
                anc_topk_feats_depth_j_padded, nn_topk_feats_depth_j_padded.transpose(-1, -2)
            )
            # Mask out correspondences of any padded points
            point_similarity_mask = torch.logical_or(
                anc_topk_mask_depth_j_padded[..., None], nn_topk_mask_depth_j_padded[..., None, :]
            )
            point_similarity_matrix.masked_fill_(point_similarity_mask, self.similarity_mask_val)
            # TODO: IF AN END-OF-BATCH RT IS SELECTED, IT CAN HAVE LESS THAN k 
            #       VALID POINTS IN IT, THUS ALLOWING PADDED POINTS TO LEAK THROUGH.
            #       NEED TO EITHER:
            #           *A: FILTER OUT RT CORRESPONDENCES WITH LESS THAN k VALID POINTS.
            #            B: SOMEHOW PICK OTHER POINTS TO COMPENSATE
            #            C: SWEEP IT UNDER THE RUG (bad idea)
 
            # Compute top-k correspondences per window, and combine
            if self.mutual:
                raise NotImplementedError
            else:
                # Just use top-k correspondences from anc side
                # NOTE: Currently this doesn't guarantee a one-to-one mapping,
                #       so the same point could be used multiple times.
                anc_points_correspondence_scores, anc_points_correspondence_indices = torch.max(point_similarity_matrix, dim=-1)
                _, nn_points_topk_correspondence_indices = torch.topk(
                    anc_points_correspondence_scores, k=correspondences_per_window, dim=-1
                )
                anc_points_topk_correspondence_indices = torch.gather(
                    anc_points_correspondence_indices, dim=-1, index=nn_points_topk_correspondence_indices
                )

            # Select top-k point correspondences
            anc_final_points = torch.gather(
                anc_topk_points_depth_j_padded, dim=-2, index=nn_points_topk_correspondence_indices[..., None].expand(-1, -1, -1, -1, 3)
            ).view(B, 2, -1, 3)[..., :self.num_correspondences[ii], :]
            nn_final_points = torch.gather(
                nn_topk_points_depth_j_padded, dim=-2, index=anc_points_topk_correspondence_indices[..., None].expand(-1, -1, -1, -1, 3)
            ).view(B, 2, -1, 3)[..., :self.num_correspondences[ii], :]
            
            # Compute spectral geometric consistency
            # TODO: CONFIRM THIS FUNC WORKS WITH PROVIDED CORRESPONDENCES
            leading_eigvec, spatial_consistency_score = batched_sgv_parallel(
                anc_final_points,
                nn_final_points,
                d_thresh=self.geometric_consistency_d_thresh[ii],
                return_spatial_consistency=True,
            )
            if self.sort_eigvec:
                leading_eigvec, sort_indices = torch.sort(leading_eigvec, dim=-1, descending=True)

            # TODO: Scale eigvec? (softmax? or just l2norm?)
            leading_eigvec_list.append(leading_eigvec)

        # Concat eigvecs and pass through MLP + sigmoid to get scores
        rerank_features = torch.concat(leading_eigvec_list, dim=-1)
        rerank_scores = self.output_mlp(rerank_features)
        rerank_scores = self.sigmoid(rerank_scores)

        # Create target labels
        targets = torch.zeros_like(rerank_scores)  # [B, 2, 1]
        targets[:, 0] = 1  # set label of positives to 1

        toc = time.perf_counter()
        log_str = f'Re-ranking time: {toc-tic:.4f}s'
        logging.debug(log_str)
        return rerank_scores, targets

    def rerank_inference(self, model_out: dict, shift_and_scale: Tensor, points: Points, *args, **kwargs):
        """
        Perform re-ranking of query and N candidates. Note that only 'octree',
        'rt', and 'rt_final_cls_attn_vals' keys are needed from HOTFormerLoc outputs.

        Args:
            model_out (dict): HOTFormerLoc output for entire (inference) batch. Assums first batch item is query, and all others are candidates.
            shift_and_scale (Tensor): (B, 4) tensor containing normalization parameters

        Returns:
            rerank_scores (Tensor)
        """
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        tic = time.perf_counter()
        # Collect batch relay tokens
        octree: OctreeT = model_out['octree']
        anc_idx = 0
        NN = octree.batch_size - 1  # first elem is query
        rt_cls_attn = model_out['rt_final_cls_attn_vals']
        if self.attn_topk is not None:
            # Split back to each depth
            rt_cls_attn_per_depth = unpad_and_split_rt(rt_cls_attn[..., None], octree)

        # Process each RT level
        leading_eigvec_list = []
        anc_nn_attn_scores_list = []
        for ii, rt_idx in enumerate(self.rerank_rt_indices):
            depth = octree.pyramid_depths[rt_idx]
            rt_depth_j = concat_and_pad_rt(model_out['rt'], octree, [depth])
            B, N, C = rt_depth_j.shape
            rt_centroids_depth_j = concat_and_pad_rt(octree.window_stats, octree, [depth])[..., :3]
            rt_mask = octree.rt_mask[depth][..., 0] == 0.  # Mask of non-padding RTs
            # Unnormalize RT centroids
            if shift_and_scale is not None:
                rt_centroids_depth_j = Normalize.batch_unnormalize(
                    rt_centroids_depth_j, shift_and_scale, mask=rt_mask
                )
            # Filter top-k attn vals for each depth
            if self.attn_topk is not None:
                rt_cls_attn_depth_j = concat_and_pad_rt(rt_cls_attn_per_depth, octree, [depth])
                if N < self.attn_topk[ii]:  # zero-pad
                    padding_size = self.attn_topk[ii] - N
                    rt_cls_attn_depth_j = F.pad(rt_cls_attn_depth_j, (0,0,0,padding_size))
                    rt_centroids_depth_j = F.pad(rt_centroids_depth_j, (0,0,0,padding_size))
                    rt_depth_j = F.pad(rt_depth_j, (0,0,0,padding_size))
                attn_topk_scores, attn_topk_indices = torch.topk(rt_cls_attn_depth_j, k=self.attn_topk[ii], dim=1, sorted=False)
                attn_topk_scores.squeeze_(-1)
                attn_topk_indices = torch.sort(attn_topk_indices, dim=1).values  # Sort indices to retain original ordering (ensures padding remains at end)
                rt_centroids_depth_j = torch.gather(rt_centroids_depth_j, dim=1, index=attn_topk_indices.expand(-1, -1, 3))
                rt_depth_j = torch.gather(rt_depth_j, dim=1, index=attn_topk_indices.expand(-1, -1, C))
                # Separate attn scores into anc and nn sets
                if self.use_attn_vals:
                    anc_attn_scores = attn_topk_scores[None, None, anc_idx].expand(-1, NN, -1)
                    nn_attn_scores = attn_topk_scores[None, anc_idx+1:]

            # Separate RTs into anc and nn sets (expand batch dim to 1 for compatibility)
            anc_rt_centroids = rt_centroids_depth_j[None, None, anc_idx]
            nn_rt_centroids = rt_centroids_depth_j[None, anc_idx+1:]
            anc_rt = rt_depth_j[None, None, anc_idx]
            nn_rt = rt_depth_j[None, anc_idx+1:]

            # Compute RT mask for anc and nn
            rt_sgv_mask = compute_rt_sgv_mask_inference(
                nn_rt, anc_idx, octree, depth
            )

            # Apply linear layer to RTs
            anc_rt = self.input_linear(anc_rt)
            nn_rt = self.input_linear(nn_rt)
            
            # Compute spectral geometric consistency
            leading_eigvec, spatial_consistency_score = batched_sgv_parallel(
                anc_rt_centroids,
                nn_rt_centroids,
                anc_rt,
                nn_rt,
                d_thresh=self.geometric_consistency_d_thresh[ii],
                mask=rt_sgv_mask,
                return_spatial_consistency=True,
            )
            if self.sort_eigvec:
                leading_eigvec, sort_indices = torch.sort(leading_eigvec, dim=-1, descending=True)

            if self.use_attn_vals:
                # Also sort attn vals with same index to ensure they correspond to the same relay tokens
                if self.sort_eigvec:
                    anc_nn_attn_scores_temp = torch.concat(
                        [torch.gather(anc_attn_scores, dim=-1, index=sort_indices),
                         torch.gather(nn_attn_scores, dim=-1, index=sort_indices)],
                        dim=-1,
                    )
                else:
                    anc_nn_attn_scores_temp = torch.concat(
                        [anc_attn_scores, nn_attn_scores], dim=-1
                    )

                anc_nn_attn_scores_list.append(anc_nn_attn_scores_temp)

            # TODO: Scale eigvec? (softmax? or just l2norm?)
            leading_eigvec_list.append(leading_eigvec)

        # Concat eigvecs and pass through MLP + sigmoid to get scores
        rerank_features = torch.concat(leading_eigvec_list, dim=-1)
        if self.use_attn_vals:
            # Add RT attn values to re-ranking features
            anc_nn_attn_scores = torch.concat(anc_nn_attn_scores_list, dim=-1)
            # TODO: Check order of attn scores. Should it be (anc_RT0, anc_RT1, pos_RT0, pos_RT1),
            #       or (anc_RT0, posRT0, anc_RT1, pos_RT1)?
            #       Currently is the latter, and order shouldn't matter for linear layers regardless
            rerank_features = torch.concat([rerank_features, anc_nn_attn_scores], dim=-1)

        rerank_scores = self.output_mlp(rerank_features)
        rerank_scores = self.sigmoid(rerank_scores)

        toc = time.perf_counter()
        log_str = f'Re-ranking time: {toc-tic:.4f}s'
        logging.debug(log_str)
        return rerank_scores

def compute_rt_sgv_mask(
    relay_tokens: Tensor, anc_indices, pos_indices, neg_indices, octree: OctreeT, depth: int, 
):
    """
    Computes mask to ignore padding relay tokens during geometric
    consistency [B, NN, N_RT, N_RT]
    """
    B, N_RT, C = relay_tokens.shape
    batch_num_rt_no_padding = octree.batch_num_rt_no_padding[depth].to(octree.device)
    anc_num_rt = batch_num_rt_no_padding[anc_indices]
    pos_num_rt = batch_num_rt_no_padding[pos_indices]
    neg_num_rt = batch_num_rt_no_padding[neg_indices]

    # Create full mask
    rt_sgv_mask = torch.ones((B, 2, N_RT, N_RT), dtype=bool, device=octree.device)
    for batch_idx in range(B):
        rt_sgv_mask[batch_idx, :, anc_num_rt[batch_idx]:, :] = False
        rt_sgv_mask[batch_idx, 0, :, pos_num_rt[batch_idx]:] = False
        rt_sgv_mask[batch_idx, 1, :, neg_num_rt[batch_idx]:] = False
    return rt_sgv_mask

def compute_rt_sgv_mask_inference(
    relay_tokens: Tensor, anc_idx: int, octree: OctreeT, depth: int, 
):
    """
    Computes mask to ignore padding relay tokens during geometric
    consistency [1, NN, N_RT, N_RT] (assumes batch size 1 for inference)
    """
    _, NN, N_RT, C = relay_tokens.shape
    batch_num_rt_no_padding = octree.batch_num_rt_no_padding[depth].to(octree.device)
    anc_num_rt = batch_num_rt_no_padding[anc_idx]
    nn_num_rt = batch_num_rt_no_padding[anc_idx+1:]

    # Create full mask
    rt_sgv_mask = torch.ones((1, NN, N_RT, N_RT), dtype=bool, device=octree.device)
    rt_sgv_mask[:, :, anc_num_rt:, :] = False
    for nn_idx in range(NN):
        rt_sgv_mask[:, :, :, nn_num_rt[nn_idx]:] = False
    return rt_sgv_mask