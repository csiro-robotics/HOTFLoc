"""
Relay token re-ranker with geometric consistency.

Ethan Griffiths (Data61, Pullenvale)
"""
from typing import List, Set, Tuple, Optional

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
    """Relay token re-ranker with geometric consistency."""
    def __init__(self, rerank_rt_indices: Tuple[int],
                 attn_topk: Tuple[int],
                 rt_dim: int,
                 geometric_consistency_d_thresh: float = 5.0,  # metres
                 sort_eigvec: bool = True):
        super().__init__()
        self.rerank_rt_indices = rerank_rt_indices
        self.attn_topk = attn_topk
        if len(attn_topk) != len(rerank_rt_indices):
            raise ValueError('`attn_topk` must have same num elems as `rerank_rt_indices`')
        self.rt_dim = rt_dim
        self.geometric_consistency_d_thresh = geometric_consistency_d_thresh
        self.sort_eigvec = sort_eigvec
        self.input_linear = nn.Linear(self.rt_dim, self.rt_dim)
        self.output_dim = sum(self.attn_topk)
        self.output_mlp = MLP(self.output_dim, self.output_dim, 1)  # TODO: different hidden sizes?
        self.sigmoid = nn.Sigmoid()

    def forward(self, model_out: dict, hard_triplets: Tuple[int], shift_and_scale: Tensor):
        """
        Perform re-ranking of query and N candidates. Note that only 'octree',
        'rt', and 'rt_final_cls_attn_vals' keys are needed from HOTFormerLoc outputs.

        Args:
            TODO:
            # query_output (dict): HOTFormerLoc output for query submap
            # nn_outputs (List[dict]): List of HOTFormerLoc outputs for each nn submap
        """
        # NOTE: Need to re-think this a little, as entire HOTFloc output is batched
        #       during training. Instead, pass a (query_indices, nn_indices) param
        #       to specify our re-ranking batch (hard mining in loss func).
        
        # TODO: Plan -- 
        #       - Get query RTs and nn RTs (add param for num levels / level idx)
        #       - Sort each by top-k attn vals (with zero-padding for safety?)
        #       - Apply linear layer + (optional) L2 norm
        #       - Compute/get RT centroids from OctreeTs
        #       - Apply SGV
        #       - Sort + scale (+ concat) eigenvectors and process with MLP + sigmoid
        #       - Return (batched) re-ranking scores

        # Collect batch relay tokens
        octree: OctreeT = model_out['octree']
        anc_indices, pos_indices, neg_indices = hard_triplets
        rt_cls_attn = model_out['rt_final_cls_attn_vals']
        if self.attn_topk is not None:
            # Split back to each depth
            rt_cls_attn_per_depth = unpad_and_split_rt(rt_cls_attn[..., None], octree)

        # Process each RT level
        leading_eigvec_list = []
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
                # TODO: finish top-k bit
                if N < self.attn_topk[ii]:  # zero-pad
                    padding_size = self.attn_topk[ii] - N
                    rt_cls_attn_depth_j = F.pad(rt_cls_attn_depth_j, (0,0,0,padding_size))
                    rt_centroids_depth_j = F.pad(rt_centroids_depth_j, (0,0,0,padding_size))
                    rt_depth_j = F.pad(rt_depth_j, (0,0,0,padding_size))
                    # TODO: Fix masking to consider added padding
                attn_topk_scores, attn_topk_indices = torch.topk(rt_cls_attn_depth_j, k=self.attn_topk[ii], dim=1, sorted=False)
                rt_centroids_depth_j = torch.gather(rt_centroids_depth_j, dim=1, index=attn_topk_indices.expand(-1, -1, 3))
                rt_depth_j = torch.gather(rt_depth_j, dim=1, index=attn_topk_indices.expand(-1, -1, C))

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
                d_thresh=self.geometric_consistency_d_thresh,
                mask=rt_sgv_mask,
                return_spatial_consistency=True,
            )
            if self.sort_eigvec:
                leading_eigvec = torch.sort(leading_eigvec, dim=-1, descending=True).values
            # TODO: Scale eigvec? (softmax? or just l2norm?)
            leading_eigvec_list.append(leading_eigvec)

        # Concat eigvecs and pass through MLP + sigmoid to get scores
        leading_eigvec_all = torch.concat(leading_eigvec_list, dim=-1)

        # MLP
        # TODO: Verify she's all good
        rerank_scores = self.output_mlp(leading_eigvec_all)
        rerank_scores = self.sigmoid(rerank_scores)

        # Create target labels
        targets = torch.zeros_like(rerank_scores)  # [B, 2]
        targets[:, 0] = 1
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