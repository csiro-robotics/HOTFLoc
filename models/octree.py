# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
#
# Adapted from https://github.com/octree-nn/octformer
# by Ethan Griffiths.
# --------------------------------------------------------

from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
import ocnn
from ocnn.octree import Octree


def pad_sequence(batch_list, fill_value: int = 0) -> torch.Tensor:
    """
    Collate list of different size tensors into a batch via padding. I found
    this implementation faster than torch.nn.utils.rnn.pad_sequence(). 
    """
    data_padded_list = []
    max_size = max([row.size(0) for row in batch_list])
    for row in batch_list:
        data_padded_list.append(
            F.pad(
                row, pad=(0, 0, 0, max_size - row.size(0)), value=fill_value
            )
        )
    data_padded = torch.stack(data_padded_list)
    return data_padded        


class OctreeT(Octree):
    """
    Octree window attention data structure adapted from
    https://github.com/octree-nn/octformer, with Hierarchical Attention (HAT)
    design inspired by https://github.com/NVlabs/FasterViT.
    """
    def __init__(self, octree: Octree, patch_size: int = 24, dilation: int = 4,
                 nempty: bool = True, max_depth: Optional[int] = None,
                 start_depth: Optional[int] = None, ct_size: int = 0, **kwargs):
        super().__init__(octree.depth, octree.full_depth)
        self.__dict__.update(octree.__dict__)

        self.patch_size = patch_size
        self.dilation = dilation  # TODO dilation as a list
        self.ct_size = ct_size
        self.nempty = nempty
        self.max_depth = max_depth or self.depth
        self.start_depth = start_depth or self.full_depth
        self.invalid_mask_value = -1e3
        assert self.start_depth > 1, "Octree not deep enough for model depth"

        self.block_num = patch_size * dilation
        self.nnum_t = self.nnum_nempty if nempty else self.nnum
        self.nnum_a = ((self.nnum_t / self.block_num).ceil() * self.block_num).int()

        num = self.max_depth + 1
        self.batch_idx = [None] * num
        self.ct_batch_idx = [None] * num
        self.batch_boundary = [None] * num
        self.batch_num_windows = [None] * num
        self.batch_window_overlap_mask = [None] * num
        self.patch_mask = [None] * num
        self.dilate_mask = [None] * num
        self.hat_window_mask = [None] * num
        self.ct_mask = [None] * num
        self.ct_init_mask  = [None] * num
        self.rel_pos = [None] * num
        self.dilate_pos = [None] * num
        self.build_t()

    def build_t(self):
        for d in range(self.start_depth, self.max_depth + 1):
            self.build_batch_idx(d)
            self.build_batch_boundary(d)
            self.build_attn_mask(d)
            self.build_ct_attn_mask(d)
            self.build_rel_pos(d)

    def build_batch_idx(self, depth: int):
        batch = self.batch_id(depth, self.nempty)
        self.batch_idx[depth] = self.patch_partition(batch, depth, self.batch_size)
        batch_ct = self.batch_idx[depth].view(-1, self.patch_size // self.ct_size)
        self.ct_batch_idx[depth] = batch_ct.min(1).values

    def build_batch_boundary(self, depth: int):
        """
        Get the boundary indices for each batch elem. Useful for separating CTs
        into batches with torch.tensor_split().
        """
        if self.ct_size == 0:
            return
        batch_nnum_cumsum = self.batch_nnum_nempty[depth].cumsum(0)
        # Add patch partition padding to last elem
        num_padded = self.nnum_a[depth] - self.nnum_t[depth]
        batch_nnum_cumsum[-1] = batch_nnum_cumsum[-1] + num_padded
        # Get idxs where batch changes
        batch_boundary_floor = batch_nnum_cumsum // self.patch_size
        # Get number of leftover points in last window of each batch
        batch_window_remainder = batch_nnum_cumsum % self.patch_size
        # Create mask for batch windows that contain overlapping batch data
        self.batch_window_overlap_mask[depth] = batch_window_remainder.masked_fill(batch_window_remainder != 0, 1)
        # Correct indices for splitting with tensor_split
        self.batch_boundary[depth] = batch_boundary_floor + self.batch_window_overlap_mask[depth]
        # Also get number of windows per batch elem, inclusive of overlap with next elem (used for torch.split)
        self.batch_num_windows[depth] = torch.diff(self.batch_boundary[depth], prepend=torch.zeros(1)).int()
        
    def build_attn_mask(self, depth: int):
        batch = self.batch_idx[depth]
        mask = batch.view(-1, self.patch_size)
        self.patch_mask[depth] = self._calc_attn_mask(mask)

        # Patch + CT mask (HAT)
        # TODO: check this works with ct_size > 1
        # NOTE: Currently, overlapping CTs are not masked out, and instead are
        #       masked so that they only attend to features from the leftmost
        #       batch element (i.e. floor of the batches they belong to)
        if self.ct_size > 0:
            # Use left-most batch idx for carrier tokens
            batch_ct_idx = mask.min(1, keepdim=True).values
            # Save mask for CT initialisation (prevents pooling erroneous features)
            self.ct_init_mask[depth] = mask != batch_ct_idx
            # Add CT to mask
            mask = F.pad(mask, pad=(self.ct_size, 0, 0, 0))
            mask[:, :self.ct_size] += batch_ct_idx  # insert CT batch idx
            # overlap_idx = self.batch_boundary[depth][self.batch_window_overlap_mask[depth] == 1] - 1
            # mask[overlap_idx, :self.ct_size] = self.batch_size + 1e4                               # MASK OUT ALL OVERLAP CTs
            # mask[overlap_idx, :self.ct_size] = mask[overlap_idx].mode(dim=1, keepdim=True).values  # KEEP OVERLAP CTs FOR BATCH ELEMENT WITH MORE DATA
            self.hat_window_mask[depth] = self._calc_attn_mask(mask)

        mask = batch.view(-1, self.patch_size, self.dilation)
        mask = mask.transpose(1, 2).reshape(-1, self.patch_size)
        self.dilate_mask[depth] = self._calc_attn_mask(mask)
    
    def build_ct_attn_mask(self, depth: int):
        """
        Compute attention mask for carrier tokens, so that attention ignores
        padding and all CTs that contain neighbouring batch information.
        """
        # NOTE: Currently, overlapping CTs are not masked out, and instead are
        #       masked so that they only attend to features from the leftmost
        #       batch element (i.e. floor of the batches they belong to)
        if self.ct_size == 0:
            return
        # TODO: make this work for ct_size > 1
        batch_num_windows_list = self.batch_num_windows[depth].tolist()
        batch_windows = self.batch_idx[depth].view(-1, self.patch_size)  # batch idx of each window
        batch_windows_split = batch_windows.split(batch_num_windows_list)
        # Pad with values higher than batch size will ever be, to ensure fill overrides batch idx per window
        batch_windows_padded = pad_sequence(batch_windows_split, fill_value=(self.batch_size + 1e4))
        # Use left-most batch idx for carrier tokens
        mask = batch_windows_padded.min(dim=2).values
        self.ct_mask[depth] = self._calc_attn_mask(mask)

        # NOTE: Below is the start of code to correct mask for overlap windows, so that batch element with higher overlap is unmasked
        # mode_max_mask = (batch_windows.max(dim=1).values == batch_windows.mode(dim=1).values)
        # torch.nonzero(mode_max_mask == False)

    def _calc_attn_mask(self, mask: torch.Tensor):
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, self.invalid_mask_value)
        return attn_mask

    def build_rel_pos(self, depth: int):
        key = self.key(depth, self.nempty)
        key = self.patch_partition(key, depth)
        x, y, z, _ = ocnn.octree.key2xyz(key, depth)
        xyz = torch.stack([x, y, z], dim=1)

        xyz = xyz.view(-1, self.patch_size, 3)
        self.rel_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

        xyz = xyz.view(-1, self.patch_size, self.dilation, 3)
        xyz = xyz.transpose(1, 2).reshape(-1, self.patch_size, 3)
        self.dilate_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

    def patch_partition(self, data: torch.Tensor, depth: int, fill_value=0):
        num = self.nnum_a[depth] - self.nnum_t[depth]
        tail = data.new_full((num,) + data.shape[1:], fill_value)
        return torch.cat([data, tail], dim=0)

    def patch_reverse(self, data: torch.Tensor, depth: int):
        return data[:self.nnum_t[depth]]
    
    def data_to_windows(self, data: torch.Tensor, depth: int, dilated_windows: bool, fill_value=0):
        """
        Reshape octree data into windows. This function applies padding and
        dilation, so just pass the octree features, depth, and whether dilated
        windows should be used.
        """
        C = data.size(-1)
        data = self.patch_partition(data, depth, fill_value)  # (N*K, C)
        if dilated_windows:  # account for dilation
            data = data.view(-1, self.patch_size, self.dilation, C).transpose(1, 2).reshape(-1, C)
        return data.view(-1, self.patch_size, C)  # (N, K, C)
    
    def windows_to_data(self, data: torch.Tensor, depth: int, dilated_windows: bool):
        """
        Reshape octree windows back into original shape. This function accounts
        for padding and dilation, so just pass the octree features, depth, and
        whether dilated windows are used.
        """
        C = data.size(-1)
        data = data.reshape(-1, C)  # (N*K, C)
        if dilated_windows:  # account for dilation
            data = data.view(-1, self.dilation, self.patch_size, C).transpose(1, 2).reshape(-1, C)
        return self.patch_reverse(data, depth)