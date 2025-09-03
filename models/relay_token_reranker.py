"""
Relay token re-ranker with geometric consistency.

Ethan Griffiths (Data61, Pullenvale)
"""
from typing import List, Set, Tuple, Optional
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataset.augmentation import Normalize
from models.octree import OctreeT
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

    def forward(self, model_out: dict, hard_triplets: Tuple, shift_and_scale: Tensor):
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
            rt_mask = octree.ct_mask[depth] == 0.  # Mask of non-padding RTs
            # Unnormalize RT centroids
            if shift_and_scale is not None:
                rt_centroids_depth_j = Normalize.batch_unnormalize(
                    rt_centroids_depth_j, shift_and_scale, mask=rt_mask[..., 0]
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
            rt_sgv_mask = self.compute_rt_sgv_mask(
                rt_depth_j, anc_indices, pos_indices, neg_indices, octree, depth
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
        targets = torch.zeros_like(rerank_scores)  # [B, 2]
        targets[:, 0] = 1  # set label of positives to 1

        toc = time.perf_counter()
        log_str = f'Re-ranking time: {toc-tic:.4f}s'
        logging.debug(log_str)
        return rerank_scores, targets

    @staticmethod
    def compute_rt_sgv_mask(
        relay_tokens, anc_indices, pos_indices, neg_indices, octree: OctreeT, depth: int
    ):
        """
        Computes mask to ignore padding relay tokens during geometric
        consistency [B, NN, NPTS, NPTS]
        """
        B, N_RT, C = relay_tokens.shape
        # Get number of non-padding RTs per batch
        rt_batch_idx_list = octree.ct_batch_idx[depth].split(octree.batch_num_windows[depth].tolist())
        num_rt_per_batch = torch.tensor([torch.count_nonzero(x == i) for i, x in enumerate(rt_batch_idx_list)])
        anc_num_rt = num_rt_per_batch[anc_indices]
        pos_num_rt = num_rt_per_batch[pos_indices]
        neg_num_rt = num_rt_per_batch[neg_indices]

        # Create full mask
        rt_sgv_mask = torch.ones((B, 2, N_RT, N_RT), dtype=bool, device=octree.device)
        for batch_idx in range(B):
            rt_sgv_mask[batch_idx, :, anc_num_rt[batch_idx]:, :] = False
            rt_sgv_mask[batch_idx, 0, :, pos_num_rt[batch_idx]:] = False
            rt_sgv_mask[batch_idx, 1, :, neg_num_rt[batch_idx]:] = False
        return rt_sgv_mask