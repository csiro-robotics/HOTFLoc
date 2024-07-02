# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
#
# Adapted for use with carrier tokens by Ethan Griffiths.
# --------------------------------------------------------

import torch
from typing import Optional

from models.octree import OctreeT


class OctreeDropPath(torch.nn.Module):
    r'''Drop paths (Stochastic Depth) per sample when applied in main path of
    residual blocks, following the logic of :func:`timm.models.layers.DropPath`.

    Args:
        drop_prob (int): The probability of drop paths.
        nempty (bool): Indicate whether the input data only contains features of
            the non-empty octree nodes or not.
        scale_by_keep (bool): Whether to scale the kept features proportionally.
        dilated_windows (bool): Whether dilation is being used.
    '''

    def __init__(self, drop_prob: float = 0.0, nempty: bool = False,
                 scale_by_keep: bool = True, dilated_windows: bool = False,
                 use_ct: bool = False):
        super().__init__()

        self.drop_prob = drop_prob
        self.nempty = nempty
        self.scale_by_keep = scale_by_keep
        self.dilated_windows = dilated_windows
        self.use_ct = use_ct

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int,
                batch_id: Optional[torch.Tensor] = None):
        r''''''

        if self.drop_prob <= 0.0 or not self.training:
            return data

        batch_size = octree.batch_size
        ndim = data.ndim
        K = data.size(1)  # for ndim = 3, 2nd dim is the window dim
        assert ndim in (2, 3), "Invalid num dimensions in input"
        keep_prob = 1 - self.drop_prob
        rnd_tensor = torch.rand(
            batch_size, 1, dtype=data.dtype, device=data.device
        )
        rnd_tensor = torch.floor(rnd_tensor + keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            rnd_tensor.div_(keep_prob)

        if batch_id is None:
            batch_id = octree.batch_id(depth, self.nempty)
            # Check if dealing with Octree batch or windowed/ct batch
            if self.use_ct:
                # Get batch id of each ct
                batch_id = octree.ct_batch_idx[depth]
            elif ndim == 3:
                if K == octree.patch_size:  # standard window attn
                    batch_id = octree.data_to_windows(
                        batch_id.unsqueeze(-1), depth, self.dilated_windows,
                        fill_value=(batch_size - 1)
                    ).squeeze(-1)
                    # NOTE: Padding is almost guaranteed to belong only to the
                    #       final batch elem (as long as num_windows >=
                    #       dilation), and it doesn't matter anyways since it is
                    #       just padding that will be getting dropped.
                else:  # HAT attn (window + CT)
                    batch_id = octree.hat_batch_idx[depth]
                    # Assume padding idx as part of last batch
                    batch_id = batch_id.minimum(torch.tensor(batch_size - 1))                    

        drop_mask = rnd_tensor[batch_id]
        output = data * drop_mask
        return output

    def extra_repr(self) -> str:
        return ('drop_prob={:.4f}, nempty={}, scale_by_keep={}').format(
                self.drop_prob, self.nempty, self.scale_by_keep)  # noqa
