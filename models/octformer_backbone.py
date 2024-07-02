# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
#
# Adapted from https://github.com/octree-nn/octformer
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import unpad_sequence
import ocnn
import dwconv

from ocnn.octree import Octree
from typing import Optional, List, Dict
from torch.utils.checkpoint import checkpoint
from models.octree import OctreeT, pad_sequence
from models.layers.mask_powernorm import MaskPowerNorm
from models.layers.octree_drop import OctreeDropPath 


def get_norm_layer(channels: int, norm_type: str = 'batchnorm'):
    """
    Return the desired normalisation layer.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batchnorm':
        norm_layer = torch.nn.BatchNorm1d(channels)
    elif norm_type == 'layernorm':
        norm_layer = torch.nn.LayerNorm(channels)
    elif norm_type == 'powernorm':
        norm_layer = MaskPowerNorm(channels)
    else:
        raise ValueError("Norm type must be either 'batchnorm' or 'layernorm'")
    return norm_layer


class MLP(torch.nn.Module):

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                out_features: Optional[int] = None, activation=torch.nn.GELU,
                drop: float = 0.0, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.fc1 = torch.nn.Linear(self.in_features, self.hidden_features)
        self.act = activation()
        self.fc2 = torch.nn.Linear(self.hidden_features, self.out_features)
        self.drop = torch.nn.Dropout(drop, inplace=True)

    def forward(self, data: torch.Tensor):
        data = self.fc1(data)
        data = self.act(data)
        data = self.drop(data)
        data = self.fc2(data)
        data = self.drop(data)
        return data


class OctreeDWConvNorm(torch.nn.Module):
    """
    Sequence of Octree DWConv, and BatchNorm/LayerNorm.
    """

    def __init__(self, in_channels: int, kernel_size: List[int] = [3],
                 nempty: bool = False, conv_norm: str = 'batchnorm'):
        super().__init__()
        self.conv = dwconv.OctreeDWConv(
            in_channels, kernel_size, nempty, use_bias=False)
        self.norm = get_norm_layer(in_channels, conv_norm)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.norm(out)
        return out


class OctreeConvNormRelu(torch.nn.Module):
    """
    Sequence of Octree Conv, BatchNorm/LayerNorm, and Relu.
    """

    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: List[int] = [3], stride: int = 1,
                nempty: bool = False, conv_norm: str = 'batchnorm'):
        super().__init__()
        self.conv = ocnn.nn.OctreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.norm = get_norm_layer(out_channels, conv_norm)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.norm(out)
        out = self.relu(out)
        return out


class OctreeDeconvNormRelu(torch.nn.Module):
    """
    Sequence of Octree Deconv, BatchNorm/LayerNorm, and Relu.
    """

    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: List[int] = [3], stride: int = 1,
                nempty: bool = False, conv_norm: str = 'batchnorm'):
        super().__init__()
        self.deconv = ocnn.nn.OctreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.norm = get_norm_layer(out_channels, conv_norm)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.deconv(data, octree, depth)
        out = self.norm(out)
        out = self.relu(out)
        return out


class RPE(torch.nn.Module):

    def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.pos_bnd = self.get_pos_bnd(patch_size)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3*self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def get_pos_bnd(self, patch_size: int):
        return int(0.8 * patch_size * self.dilation**0.5)

    def xyz2idx(self, xyz: torch.Tensor):
        mul = torch.arange(3, device=xyz.device) * self.rpe_num
        xyz = xyz.clamp(-self.pos_bnd, self.pos_bnd)
        idx = xyz + (self.pos_bnd + mul)
        return idx

    def forward(self, xyz):
        idx = self.xyz2idx(xyz)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

    def extra_repr(self) -> str:
        return 'num_heads={}, pos_bnd={}, dilation={}'.format(
                        self.num_heads, self.pos_bnd, self.dilation)  # noqa


class OctreeAttention(torch.nn.Module):

    def __init__(self, dim: int, patch_size: int, num_heads: int,
                qkv_bias: bool = True, qk_scale: Optional[float] = None,
                attn_drop: float = 0.0, proj_drop: float = 0.0,
                dilation: int = 1, ct_per_window: int = 0, use_rpe: bool = True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.ct_per_window = ct_per_window
        self.use_rpe = use_rpe
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)

        # NOTE: self.rpe is not used in the original experiments of my paper. When
        # releasing the code, I added self.rpe because I observed that it could
        # stablize the training process and improve the performance on ScanNet by
        # 0.3 to 0.5; on the other datasets, the improvements are more marginal. So
        # it is not indispensible, and can be removed by setting `use_rpe` as False.
        self.rpe = RPE(patch_size, num_heads, dilation) if use_rpe else None

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        H = self.num_heads
        K = self.patch_size
        C = self.dim
        D = self.dilation
        G = self.ct_per_window

        # TODO: do patch partition and reshaping outside of attn block
        # # patch partition
        # data = octree.patch_partition(data, depth)
        # TODO: CTs need to be considered in patch mask
        if D > 1:  # dilation
            rel_pos = octree.dilate_pos[depth]
            mask = octree.dilate_mask[depth]
            # data = data.view(-1, K, D, C).transpose(1, 2).reshape(-1, C)
        else:
            rel_pos = octree.rel_pos[depth]
            if G > 0:  # get correct mask for HAT attention
                mask = octree.hat_window_mask[depth]
            else:
                mask = octree.patch_mask[depth]
        # data = data.view(-1, K, C)

        # qkv
        qkv = self.qkv(data).reshape(-1, K+G, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # (N, H, K+G, C')
        q = q * self.scale

        # attn
        attn = q @ k.transpose(-2, -1)        # (N, H, K+G, K+G)
        attn = self.apply_rpe(attn, rel_pos)  # (N, H, K+G, K+G)
        attn = attn + mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # data = (attn @ v).transpose(1, 2).reshape(-1, C)  # (N*(K+G), C)
        data = (attn @ v).transpose(1, 2).reshape(-1, K+G, C)  # (N, K+G, C)

        # TODO: check if this can be done outside of attn, or if proj requires it before
        # # patch reverse
        # if D > 1:  # dilation
        #     data = data.view(-1, D, K, C).transpose(1, 2).reshape(-1, C)
        # data = octree.patch_reverse(data, depth)

        # ffn
        data = self.proj(data)
        data = self.proj_drop(data)
        return data

    def apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            # TODO: pad RPE for CTs
            if self.ct_per_window > 0:
                F.pad(rpe, (self.ct_per_window,0)).contiguous()
            attn = attn + self.rpe(rel_pos)
        return attn

    def extra_repr(self) -> str:
        return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
                        self.dim, self.patch_size, self.num_heads, self.dilation)  # noqa


class CTAttention(torch.nn.Module):
    # TODO: implement this, check if I need to precompute max num carrier tokens?

    def __init__(self, dim: int, patch_size: int, num_heads: int,
                qkv_bias: bool = True, qk_scale: Optional[float] = None,
                attn_drop: float = 0.0, proj_drop: float = 0.0,
                ct_per_window: int = 0,
                use_rpe: bool = True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.ct_per_window = ct_per_window
        self.num_heads = num_heads
        self.use_rpe = use_rpe  # TODO: implement RPE
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)

        """ NOTE: RPE table is currently constructed using relative pos of
        octree nodes, but this is not so easy to do for CTs. Need to determine
        how to do this, maybe using Swinv2 continuous RPE.
        """
        # self.rpe = RPE(patch_size, num_heads, dilation) if use_rpe else None

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        B = octree.batch_size
        H = self.num_heads
        C = self.dim

        # split CTs into batches for each batch elem, padded to size of largest batch
        batch_num_windows = octree.batch_num_windows[depth]
        data = data.split(batch_num_windows.tolist())        
        ### OLD METHOD ###
        # batch_boundary = octree.batch_boundary[depth]
        # data = data.tensor_split(batch_boundary)[:octree.batch_size]
        ##################
        data = pad_sequence(data)
        
        # get CT attn mask
        ct_mask = octree.ct_mask[depth]
        
        # qkv
        qkv = self.qkv(data).reshape(B, -1, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # (B, H, K, C')
        q = q * self.scale
        
        # attn
        attn = q @ k.transpose(-2, -1)        # (B, H, K, K)
        # attn = self.apply_rpe(attn, rel_pos)  # TODO: implement RPE
        attn = attn + ct_mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        data = (attn @ v).transpose(1, 2).reshape(B, -1, C)  # (B, K, C)

        # Undo padding
        data = torch.cat(unpad_sequence(data, batch_num_windows, batch_first=True))

        # ffn
        data = self.proj(data)
        data = self.proj_drop(data)
        return data

    def apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            attn = attn + self.rpe(rel_pos)
        return attn

    def extra_repr(self) -> str:
        return 'dim={}, ct_size={}, num_heads={}, dilation={}'.format(
                        self.dim, self.ct_per_window, self.num_heads, self.dilation)  # noqa


class OctFormerBlock(torch.nn.Module):
    """
    Octree Transformer Block adapted from https://github.com/octree-nn/octformer,
    with Hierarchical Attention (HAT) design inspired by
    https://github.com/NVlabs/FasterViT.
    """
    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                activation: torch.nn.Module = torch.nn.GELU, use_ct: bool = False,
                ct_size: int = 1, disable_RPE: bool = False,
                conv_norm: str = 'batchnorm', **kwargs):
        super().__init__()
        self.use_ct = use_ct
        dilation = 1 if self.use_ct else dilation        
        self.dilated_windows = dilation > 1
        ct_per_window = ct_size if self.use_ct else 0  # track number of carrier tokens per window
        """ NOTE: Dilation is disabled when using carrier tokens, as it is
        likely redundant to use both (and carrier tokens for dilated windows
        does not make sense).
        """
        # TODO: Add per-channel learnable scalars to attn and ffn (gamma from FasterViT)
        self.norm1 = torch.nn.LayerNorm(dim)
        # TODO: alter octree attention for HAT
        self.attention = OctreeAttention(dim, patch_size, num_heads, qkv_bias,
                                         qk_scale, attn_drop, proj_drop, dilation,
                                         ct_per_window=ct_per_window,
                                         use_rpe=(not disable_RPE))
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.drop_path = OctreeDropPath(drop_path, nempty,
                                        dilated_windows=self.dilated_windows)
        self.cpe = OctreeDWConvNorm(dim, nempty=nempty, conv_norm=conv_norm)
        if self.use_ct:  # carrier token attention layers
            # TODO: may need another PE here as any conv-based ones may not work (since CTs are outside of octree structure)
            # self.ct_cpe = OctreeDWConvNorm(dim, nempty=nempty, conv_norm=conv_norm)
            self.ct_norm1 = torch.nn.LayerNorm(dim)
            self.ct_attention = CTAttention(dim, patch_size, num_heads, qkv_bias,
                                            qk_scale, attn_drop, proj_drop,
                                            ct_per_window,
                                            use_rpe=(not disable_RPE))
            self.ct_norm2 = torch.nn.LayerNorm(dim)
            self.ct_mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
            self.ct_drop_path = OctreeDropPath(drop_path, nempty, use_ct=True)

    def forward(self, data: torch.Tensor, carrier_tokens: torch.Tensor, octree: OctreeT, depth: int):
        ct = carrier_tokens
        # Apply conditional positional encoding
        data = self.cpe(data, octree, depth) + data
        # Pad batch and reshape into windows
        data = octree.data_to_windows(  # (N, K, C)
            data, depth, dilated_windows=self.dilated_windows
        )
        if self.use_ct:
            # Do global attention via carrier tokens
            # TODO: carrier token PE + attention + mlp, then concat with window tokens
            # ct = self.hat_cpe(ct, octree, depth) + ct
            ct_attn = self.ct_attention(self.ct_norm1(ct), octree, depth)
            # TODO: ensure drop_path is working correctly (drop_path needs actual batch size as first dim)
            ct = ct + self.ct_drop_path(ct_attn, octree, depth)
            ct_ffn = self.ct_mlp(self.ct_norm2(ct))
            ct = ct + self.ct_drop_path(ct_ffn, octree, depth)
            
            # concatenate carrier tokens to the windowed tokens
            data = torch.cat((ct.unsqueeze(1), data), dim=1)

        # TODO: cat local windows
        attn = self.attention(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        ffn = self.mlp(self.norm2(data))
        data = data + self.drop_path(ffn, octree, depth)

        # TODO: Split CTs from window tokens
        if self.use_ct:
            ct, data = data.split([], dim=1)
        
        # Unpad batch and restore original data shape
        data = octree.windows_to_data(
            data, depth, dilated_windows=self.dilated_windows
        )

        # TODO: on last block, propagate carrier token info to the feature map
        return data, ct


class TokenInitialiser(torch.nn.Module):
    """
    Carrier token Initialiser based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """
    def __init__(self, dim, patch_size, nempty, conv_norm, ct_size=1):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            patch_size: patch size.
            nempty: only compute on non-empty octree leaf nodes.
            conv_norm: type of normalisation to use after the conv layer.
            ct_size: number of carrier tokens per local window.
        """
        super().__init__()
        self.cpe = OctreeDWConvNorm(dim, nempty=nempty, conv_norm=conv_norm)
        """ NOTE: Currently, because of how octree windows are constructed,
        consecutive batch elements can have an octree window with elements
        from both batches. This means avgpooled features for 'leaky' windows
        will contain features from 2 batch elements, and not be valid. The
        only way to prevent this (that I can tell) is to redo the OCNN batch
        implementation to include padding around each batch element. Instead,
        I opt to ignore 'leaky' window features during global attention. This
        should be fine most of the time as a max of 2 windows will be ignored
        per batch, typically out of 100s, but isn't the optimal solution.
        """
        # Pool the features in each octree window, without considering surrounding features
        assert patch_size % ct_size == 0, "Currently, patch_size must be divisible by ct_size"
        # self.pool = torch.nn.AvgPool1d(kernel_size=patch_size//ct_size)
        self.patch_size = patch_size
        self.dim = dim
        self.ct_size = ct_size

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        K = self.patch_size
        C = self.dim
        G = self.ct_size
        
        data = self.cpe(data, octree, depth)
        data = octree.patch_partition(data, depth)
        # Reshape to avg pool over spatial dimension
        # data = data.permute(1, 0).unsqueeze(0)
        # ct = self.pool(data)
        # ct = ct.squeeze(0).permute(1, 0)  # (batch_CTs, C)

        # Reshape to windows, and mask out ignored values as NaN
        # TODO: Make this work with ct_size > 1
        data = data.view(-1, K//G, C)
        mask = octree.ct_init_mask[depth].unsqueeze(-1)
        data = data.masked_fill(mask, torch.nan)
        # AvgPool1D can't handle NaNs, so use nanmean() instead
        ct = torch.nanmean(data, dim=1)
        assert(not torch.any(ct.isnan())), "NaN propagated during CT init, check code"        
        return ct

class OctFormerStage(torch.nn.Module):

    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
                disable_RPE: bool = False, use_ct: bool = False, ct_size: int = 1,
                grad_checkpoint: bool = True, num_blocks: int = 2,
                conv_norm: str = 'batchnorm', octformer_block=OctFormerBlock,
                **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        self.grad_checkpoint = grad_checkpoint
        self.use_ct = use_ct
        self.interval = interval  # normalisation interval
        self.num_norms = (num_blocks - 1) // self.interval

        self.blocks = torch.nn.ModuleList([octformer_block(
                dim=dim, num_heads=num_heads, patch_size=patch_size,
                dilation=1 if (i % 2 == 0) else dilation,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=proj_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                nempty=nempty, activation=activation, disable_RPE=disable_RPE,
                use_ct=use_ct, ct_size=ct_size,
                conv_norm=conv_norm) for i in range(num_blocks)])
        if self.use_ct:
            self.global_tokeniser = TokenInitialiser(dim,
                                                     patch_size=patch_size,
                                                     nempty=nempty,
                                                     conv_norm=conv_norm,
                                                     ct_size=ct_size)
        # self.norms = torch.nn.ModuleList([
        #     torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        ct = self.global_tokeniser(data, octree, depth) if self.use_ct else None
        for i in range(self.num_blocks):
            if self.grad_checkpoint and self.training:
                data, ct = checkpoint(self.blocks[i], data, ct, octree, depth, use_reentrant=False)  # disable reentrant to fix error with no_grad?
            else:
                data, ct = self.blocks[i](data, ct, octree, depth)
            # if i % self.interval == 0 and i != 0:
            #   data = self.norms[(i - 1) // self.interval](data)
        return data


class PatchEmbed(torch.nn.Module):

    def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 2,
                nempty: bool = True, downsample_input_embeddings: bool = True,
                conv_norm: str = 'batchnorm', **kwargs):
        super().__init__()
        self.num_stages = num_down
        self.delta_depth = -num_down
        self.downsample_input_embeddings = downsample_input_embeddings

        if self.downsample_input_embeddings:
            channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]
            self.convs = torch.nn.ModuleList([OctreeConvNormRelu(
                in_channels if i == 0 else channels[i], channels[i], kernel_size=[3],
                stride=1, nempty=nempty, conv_norm=conv_norm) for i in range(self.num_stages)])
            self.downsamples = torch.nn.ModuleList([OctreeConvNormRelu(
                channels[i], channels[i+1], kernel_size=[2], stride=2, nempty=nempty, conv_norm=conv_norm)
                for i in range(self.num_stages)])
            self.proj = OctreeConvNormRelu(
                channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty, conv_norm=conv_norm)
        else:
            self.convs = torch.nn.ModuleList([OctreeConvNormRelu(
                in_channels if i == 0 else dim, dim, kernel_size=[3],
                stride=1, nempty=nempty, conv_norm=conv_norm) for i in range(self.num_stages)])

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        if self.downsample_input_embeddings:
            for i in range(self.num_stages):
                depth_i = depth - i
                data = self.convs[i](data, octree, depth_i)
                data = self.downsamples[i](data, octree, depth_i)
            data = self.proj(data, octree, depth_i - 1)
        else:
            for i in range(self.num_stages):
                data = self.convs[i](data, octree, depth)
        return data


class Downsample(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: List[int] = [2], nempty: bool = True,
                conv_norm: str = 'batchnorm'):
        super().__init__()
        self.conv = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size,
                                    stride=2, nempty=nempty, use_bias=True)		
        self.norm = get_norm_layer(out_channels, conv_norm)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.conv(data, octree, depth)
        data = self.norm(data)
        return data


class OctFormerBase(torch.nn.Module):

    def __init__(self, in_channels: int,
                channels: List[int] = [96, 192, 384, 384],
                num_blocks: List[int] = [2, 2, 18, 2],
                num_heads: List[int] = [6, 12, 24, 24],
                ct_layers: List[bool] = [False, False, False, False],
                patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,
                nempty: bool = True, stem_down: int = 2, ct_size: int = 1,
                grad_checkpoint: bool = True, 
                downsample_input_embeddings: bool = True,
                disable_RPE: bool = False, conv_norm: str = 'batchnorm', **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.dilation = dilation
        self.nempty = nempty
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        self.downsample_input_embeddings = downsample_input_embeddings
        ct_size = ct_size if any(ct_layers) else 0
        self.ct_size = ct_size
        # Stochastic depth per block
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty, downsample_input_embeddings, conv_norm)
        self.layers = torch.nn.ModuleList([OctFormerStage(
                dim=channels[i], num_heads=num_heads[i], patch_size=patch_size,
                drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i+1])],
                dilation=dilation, nempty=nempty, disable_RPE=disable_RPE,
                grad_checkpoint=grad_checkpoint, num_blocks=num_blocks[i],
                conv_norm=conv_norm, use_ct=ct_layers[i], ct_size=ct_size)
                for i in range(self.num_stages)])
        self.downsamples = torch.nn.ModuleList([Downsample(
                channels[i], channels[i + 1], kernel_size=[2],
                nempty=nempty, conv_norm=conv_norm) for i in range(self.num_stages - 1)])

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.patch_embed(data, octree, depth)
        if self.downsample_input_embeddings:
            depth = depth - self.stem_down   # current octree depth
        octree = OctreeT(octree, self.patch_size, self.dilation, self.nempty,
                         max_depth=depth, start_depth=depth-self.num_stages+1,
                         ct_size=self.ct_size)
        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.layers[i](data, octree, depth_i)
            features[depth_i] = data
            if i < self.num_stages - 1:
                data = self.downsamples[i](data, octree, depth_i)
        return features


class FPNHeader(torch.nn.Module):

    def __init__(self, channels: List[int], fpn_channel: int, nempty: bool,
                num_top_down: int = 1, conv_norm: str = 'batchnorm'):
        super().__init__()
        self.num_top_down = num_top_down
        self.conv1x1 = torch.nn.ModuleList()  # lateral connections
        self.up_conv = torch.nn.ModuleList()  # top-down transposed convolutions

        # Generate top-down path
        for i in range(self.num_top_down):
            self.conv1x1.append(ocnn.modules.Conv1x1(
                channels[-1 - i], fpn_channel, use_bias=True))
            self.up_conv.append(OctreeDeconvNormRelu(
                fpn_channel, fpn_channel, kernel_size=[2],
                stride=2, nempty=nempty, conv_norm=conv_norm))

        # Final lateral connection from Conv block 1 or above
        self.conv1x1.append(torch.nn.Linear(channels[-1 - self.num_top_down], fpn_channel))

    def forward(self, features: Dict[int, torch.Tensor], octree: Octree):
        depth = min(features.keys())
        output_depth = depth + self.num_top_down

        # Top-down pass
        data = self.conv1x1[0](features[depth])
        for i in range(self.num_top_down):      
            data = self.up_conv[i](data, octree, depth + i)
            data = data + self.conv1x1[i + 1](features[depth + i + 1])
        
        return data, output_depth


class OctFormer(torch.nn.Module):
    """
    Octformer class adapted from https://github.com/octree-nn/octformer,
    with Hierarchical Attention (HAT) design inspired by
    https://github.com/NVlabs/FasterViT. Includes an additional FPN for
    upsampling features to use in aggregating global descriptors.
    """

    def __init__(self, in_channels: int,
                channels: List[int] = [96, 192, 384, 384],
                num_blocks: List[int] = [2, 2, 6, 2],  # default to OctFormer-small, with 6 instead of 18 blocks in 3rd stage (~20M vs ~40M params)
                num_heads: List[int] = [6, 12, 24, 24],
                ct_layers: List[bool] = [False, False, False, False],
                patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,  # NOTE: disable drop path to ensure multistage backprop is not affected? (might only be dropout that affects it)
                nempty: bool = True, stem_down: int = 2, ct_size: int = 1,
                num_top_down: int = 2, fpn_channel: int = 168,
                grad_checkpoint: bool = True, downsample_input_embeddings: bool = True,
                disable_RPE: bool = False, conv_norm: str = 'batchnorm', **kwargs):
        """
        Args:
            in_channels: Number of input channels, typically 3 if only using x,y,z information.
            channels: List containing number of feature channels per stage.
            num_blocks: List containing number of OctFormer blocks per stage.
            num_heads: List containing number of attention heads per stage, defaults to channel_size//16.
            ct_layers: List of booleans indicating which stages should use carrier tokens (ct).
            patch_size: Size of local attention patches/windows, constructed using z-order curve traversal. 
            dilation: Dilation amount for Octree attention
            drop_path: Stochastic depth probability (this is the max value stochastic depth scales to).
            nempty: Boolean indicating if only non-empty octants should be used (set True for sparse operation). 
            stem_down: Number of conv layers to use for input token embeddings, which corresponds to 2**stem_down downsampling factor.
            ct_size: Size of carrier tokens, note that patch_size must be divisible by this.
            num_top_down: Number of top-down layers in FPN. Output features will be at Octree depth d = octree_depth - stem_down - (num_stages - 1) + num_top_down.
            fpn_channel: Number of channels in FPN top-down branch. For now, set this equal to number of channels used in Pooling (i.e. output_dim param).
            grad_checkpoint: Use gradient checkpoint to save memory, at cost of extra computation time.
            downsample_input_embeddings: Do downsampling in input conv embedding.
            disable_RPE: Disable RPE during self-attention.
            conv_norm: Type of normalisation used after convolution layers. Valid params are in ['batchnorm', 'layernorm', 'powernorm'].
        """
        super().__init__()
        self.backbone = OctFormerBase(
            in_channels, channels, num_blocks, num_heads, ct_layers, 
            patch_size, dilation, drop_path, nempty, stem_down, ct_size,
            grad_checkpoint, downsample_input_embeddings, disable_RPE, conv_norm
        )
        self.head = FPNHeader(channels, fpn_channel, nempty, num_top_down, conv_norm)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        features = self.backbone(data, octree, depth)
        output, output_depth = self.head(features, octree)
        return output, output_depth
