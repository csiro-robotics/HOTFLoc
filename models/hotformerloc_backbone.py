# --------------------------------------------------------
# HOTFormerLoc: Hierarchical Octree Transformer for
# Ground-Aerial Lidar Place Recognition in Natural
# Environments
#
# Adapted from https://github.com/octree-nn/octformer by
# Ethan Griffiths (Data61, Pullenvale)
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import unpad_sequence
import ocnn

from ocnn.octree import Octree
from typing import Optional, List, Dict
from torch.utils.checkpoint import checkpoint
from models.octree import OctreeT, pad_sequence
from models.layers.octformer_layers import get_norm_layer, MLP, \
    OctreeConvNormRelu, OctreeDeconvNormRelu, CPE, RPE, ADaPE, OctreeDropPath


class OctreeAttention(torch.nn.Module):

    def __init__(self, dim: int, patch_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 dilation: int = 1, rt_per_window: int = 0, use_rpe: bool = True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.rt_per_window = rt_per_window
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
        G = self.rt_per_window

        if D > 1:  # dilation
            rel_pos = octree.dilate_pos[depth]
            mask = octree.dilate_mask[depth]
        else:
            rel_pos = octree.rel_pos[depth]
            if G > 0:  # get correct mask for HAT attention
                mask = octree.hat_window_mask[depth]
            else:
                mask = octree.patch_mask[depth]

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
        data = (attn @ v).transpose(1, 2).reshape(-1, K+G, C)  # (N, K+G, C)

        # ffn
        data = self.proj(data)
        data = self.proj_drop(data)
        return data

    def apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            rpe = self.rpe(rel_pos)
            if self.rt_per_window > 0:
                # Pad RPE for RTs (assume no relative pos for RTs)
                rpe = F.pad(rpe, (self.rt_per_window, 0, self.rt_per_window, 0))
            attn = attn + rpe
        return attn

    def extra_repr(self) -> str:
        return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
                        self.dim, self.patch_size, self.num_heads, self.dilation)  # noqa


class CTAttention(torch.nn.Module):
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

        # NOTE: RPE table is currently constructed using relative pos of
        #       octree nodes, but this is not so easy to do for CTs. Need to
        #       find another solution.
        # self.rpe = RPE(patch_size, num_heads) if use_rpe else None

    def forward(self, carrier_tokens: torch.Tensor, octree: OctreeT, depth: int):
        B = octree.batch_size
        H = self.num_heads
        C = self.dim
        ct = carrier_tokens

        # split CTs into batches for each batch elem, padded to size of largest batch
        batch_num_windows = octree.batch_num_windows[depth]
        ct = ct.split(batch_num_windows.tolist())
        ### OLD METHOD ###
        # batch_boundary = octree.batch_boundary[depth]
        # data = data.tensor_split(batch_boundary)[:octree.batch_size]
        ##################
        ct = pad_sequence(ct)

        # get CT attn mask
        ct_mask = octree.ct_mask[depth]

        # qkv
        qkv = self.qkv(ct).reshape(B, -1, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # (B, H, K, C')
        q = q * self.scale

        # attn
        attn = q @ k.transpose(-2, -1)        # (B, H, K, K)
        # attn = self.apply_rpe(attn, rel_pos)  # TODO: implement RPE
        attn = attn + ct_mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        ct = (attn @ v).transpose(1, 2).reshape(B, -1, C)  # (B, K, C)

        # Undo padding
        ct = torch.cat(unpad_sequence(ct, batch_num_windows, batch_first=True))

        # ffn
        ct = self.proj(ct)
        ct = self.proj_drop(ct)
        return ct

    def apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            attn = attn + self.rpe(rel_pos)
        return attn

    def extra_repr(self) -> str:
        return 'dim={}, ct_size={}, num_heads={}'.format(
                    self.dim, self.ct_per_window, self.num_heads)


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
                 ct_size: int = 1, ct_propagation: bool = False,
                 ct_propagation_scale: Optional[float] = None,
                 use_ADaPE: bool = False, disable_RPE: bool = False,
                 conv_norm: str = 'batchnorm', last: bool = False,
                 layer_scale: Optional[float] = None, xcpe: bool = False,
                 **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.use_ct = use_ct
        # self.use_ADaPE = use_ADaPE,
        self.dim = dim
        # NOTE: Dilation is disabled when using carrier tokens, as it is
        #       likely redundant to use both (and carrier tokens for dilated
        #       windows does not make sense).
        dilation = 1 if self.use_ct else dilation
        self.dilated_windows = dilation > 1
        ct_per_window = ct_size if self.use_ct else 0  # track number of carrier tokens per window
        self.ct_per_window = ct_per_window
        self.last = last
        self.ct_propagation = ct_propagation
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        use_ct_propagation_scale = ct_propagation_scale is not None and type(ct_propagation_scale) in [int, float]
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attention = OctreeAttention(dim, patch_size, num_heads, qkv_bias,
                                         qk_scale, attn_drop, proj_drop, dilation,
                                         rt_per_window=ct_per_window,
                                         use_rpe=(not disable_RPE))
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.drop_path = OctreeDropPath(drop_path, nempty,
                                        dilated_windows=self.dilated_windows)
        self.cpe = CPE(dim, nempty=nempty, conv_norm=conv_norm, xcpe=xcpe)
        # Learnable per-channel scale multiplier, originally proposed by
        # https://arxiv.org/pdf/2103.17239
        self.gamma1 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma2 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

        if not self.use_ct:  # carrier token attention layers
            return
        # NOTE: No longer using ADaPE per octformer block
        # if self.use_ADaPE:
        #     self.ct_adape = ADaPE(dim, activation)
        self.ct_norm1 = torch.nn.LayerNorm(dim)
        self.ct_attention = CTAttention(dim, patch_size, num_heads, qkv_bias,
                                        qk_scale, attn_drop, proj_drop,
                                        ct_per_window,
                                        use_rpe=(not disable_RPE))
        self.ct_norm2 = torch.nn.LayerNorm(dim)
        self.ct_mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.ct_drop_path = OctreeDropPath(drop_path, nempty, use_ct=True)
        self.ct_gamma1 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.ct_gamma2 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        
        if not (self.last and self.ct_propagation):
            return
        self.upsampler = torch.nn.Upsample(scale_factor=patch_size//ct_per_window, mode='nearest')
        # Just use a scalar multiplier for CT propagation scaling, which
        # prevents 'blurring' local features with CT features
        self.ct_gamma_propagate = torch.nn.Parameter(torch.tensor(ct_propagation_scale)) if use_ct_propagation_scale else 1

    def forward(self, data: torch.Tensor, carrier_tokens: torch.Tensor, octree: OctreeT, depth: int):
        K = self.patch_size
        C = self.dim
        G = self.ct_per_window
        ct = carrier_tokens
        
        # Apply conditional positional encoding
        data = data + self.cpe(data, octree, depth)
        # Pad batch and reshape into windows
        data = octree.data_to_windows(  # (N, K, C)
            data, depth, dilated_windows=self.dilated_windows
        )
        # Do global attention via carrier tokens
        if self.use_ct:
            # NOTE: No longer using ADaPE per octformer block
            # if self.use_ADaPE:
            #     ct = ct + self.ct_adape(octree, depth)
            ct_attn = self.ct_gamma1 * self.ct_attention(self.ct_norm1(ct), octree, depth)
            ct = ct + self.ct_drop_path(ct_attn, octree, depth)
            ct_ffn = self.ct_gamma2 * self.ct_mlp(self.ct_norm2(ct))
            ct = ct + self.ct_drop_path(ct_ffn, octree, depth)
            # Concatenate carrier tokens with window tokens
            data = torch.cat((ct.unsqueeze(1), data), dim=1)

        attn = self.gamma1 * self.attention(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        ffn = self.gamma2 * self.mlp(self.norm2(data))
        data = data + self.drop_path(ffn, octree, depth)

        # Split CTs from window tokens
        if self.use_ct:
            ct, data = data.split([G, K], dim=1)
            ct = ct.squeeze(1)
        # Unpad batch and restore original data shape
        data = octree.windows_to_data(
            data, depth, dilated_windows=self.dilated_windows
        )        
        # On last block, propagate carrier token features to local feature map
        if self.last and self.use_ct and self.ct_propagation:
            # TODO: Make this work with ct_size > 1
            mask = octree.ct_init_mask[depth].unsqueeze(-1)
            ct_upsampled = ct.unsqueeze(0).transpose(1, 2)
            ct_upsampled = self.upsampler(ct_upsampled).transpose(1,2).squeeze(0)
            ct_upsampled = ct_upsampled.view(-1, K//G, C)
            # Mask out padded and overlap CTs
            ct_upsampled = ct_upsampled.masked_fill(mask, value=0).view(-1, C)
            ct_upsampled = octree.patch_reverse(ct_upsampled, depth)
            data = data + self.ct_gamma_propagate*ct_upsampled
        return data, ct


class HOTFormerBlock(torch.nn.Module):
    """
    Hierarchical Octree Transformer Block adapted from
    https://github.com/octree-nn/octformer, with Hierarchical Attention (HAT)
    design loosely inspired by https://github.com/NVlabs/FasterViT.
    """
    
    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                 dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, rt_size: int = 1,
                 rt_propagation: bool = False,
                 rt_propagation_scale: Optional[float] = None,
                 disable_RPE: bool = False, conv_norm: str = 'batchnorm',
                 last: bool = False, layer_scale: Optional[float] = None,
                 xcpe: bool = False, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        # NOTE: Dilation is disabled when using carrier tokens, as it is
        #       likely redundant to use both (and carrier tokens for dilated
        #       windows does not make sense).
        dilation = 1
        self.dilated_windows = dilation > 1
        rt_per_window = rt_size  # track number of carrier tokens per window
        self.rt_per_window = rt_per_window
        self.last = last
        self.rt_propagation = rt_propagation
        use_layer_scale = (
            layer_scale is not None and type(layer_scale) in [int, float]
        )
        use_rt_propagation_scale = (
            rt_propagation_scale is not None
            and type(rt_propagation_scale) in [int, float]
        )
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attention = OctreeAttention(dim, patch_size, num_heads, qkv_bias,
                                         qk_scale, attn_drop, proj_drop, dilation,
                                         rt_per_window=rt_per_window,
                                         use_rpe=(not disable_RPE))
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.drop_path = OctreeDropPath(drop_path, nempty,
                                        dilated_windows=self.dilated_windows)
        self.cpe = CPE(dim, nempty=nempty, conv_norm=conv_norm, xcpe=xcpe)
        # Learnable per-channel scale multiplier, originally proposed by
        # https://arxiv.org/pdf/2103.17239
        self.gamma1 = torch.nn.Parameter(
            layer_scale * torch.ones(dim)
        ) if use_layer_scale else 1
        self.gamma2 = torch.nn.Parameter(
            layer_scale * torch.ones(dim)
        ) if use_layer_scale else 1

        if not (self.last and self.rt_propagation):
            return
        self.upsampler = torch.nn.Upsample(
            scale_factor=patch_size//rt_per_window, mode='nearest'
        )
        # Just use a scalar multiplier for CT propagation scaling, which
        # prevents 'blurring' local features with CT features
        self.rt_gamma_propagate = torch.nn.Parameter(
            torch.tensor(rt_propagation_scale)
        ) if use_rt_propagation_scale else 1

    def forward(self, data: torch.Tensor, relay_tokens: torch.Tensor, octree: OctreeT, depth: int):
        K = self.patch_size
        C = self.dim
        G = self.rt_per_window
        rt = relay_tokens
        
        # Apply conditional positional encoding
        data = data + self.cpe(data, octree, depth)
        # Pad batch and reshape into windows
        data = octree.data_to_windows(  # (N, K, C)
            data, depth, dilated_windows=self.dilated_windows
        )        
        # Concatenate relay tokens with window tokens
        data = torch.cat((rt.unsqueeze(1), data), dim=1)

        # Pass through transformer
        attn = self.gamma1 * self.attention(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        ffn = self.gamma2 * self.mlp(self.norm2(data))
        data = data + self.drop_path(ffn, octree, depth)

        # Split RTs from window tokens
        rt, data = data.split([G, K], dim=1)
        rt = rt.squeeze(1)
        # Unpad batch and restore original data shape
        data = octree.windows_to_data(
            data, depth, dilated_windows=self.dilated_windows
        )        
        # On last block, propagate relay token features to local feature map
        if self.last and self.rt_propagation:
            # TODO: Make this work with rt_size > 1
            mask = octree.ct_init_mask[depth].unsqueeze(-1)
            rt_upsampled = rt.unsqueeze(0).transpose(1, 2)
            rt_upsampled = self.upsampler(rt_upsampled).transpose(1,2).squeeze(0)
            rt_upsampled = rt_upsampled.view(-1, K//G, C)
            # Mask out padded and overlap CTs
            rt_upsampled = rt_upsampled.masked_fill(mask, value=0).view(-1, C)
            rt_upsampled = octree.patch_reverse(rt_upsampled, depth)
            data = data + self.rt_gamma_propagate*rt_upsampled
        return data, rt


class RelayTokenTransformerBlock(torch.nn.Module):
    """
    Relay token transformer block. Takes multi-scale relay tokens and computes
    global attention.
    """
    
    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0,
                 nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU,
                 rt_size: int = 1, use_ADaPE: bool = False,
                 conv_norm: str = 'batchnorm',
                 layer_scale: Optional[float] = None, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        # self.use_ADaPE = use_ADaPE,
        self.dim = dim
        rt_per_window = rt_size  # track number of carrier tokens per window
        self.rt_per_window = rt_per_window
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]

        # NOTE: No longer using ADaPE per octformer block
        # if self.use_ADaPE:
        #     self.ct_adape = ADaPE(dim, activation)
        self.ct_norm1 = torch.nn.LayerNorm(dim)
        self.ct_attention = CTAttention(dim, patch_size, num_heads, qkv_bias,
                                        qk_scale, attn_drop, proj_drop,
                                        rt_per_window,)
        self.ct_norm2 = torch.nn.LayerNorm(dim)
        self.ct_mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.ct_drop_path = OctreeDropPath(drop_path, nempty, use_ct=True)
        self.ct_gamma1 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.ct_gamma2 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        

    def forward(self, relay_token_dict: torch.Tensor, octree: OctreeT):
        K = self.patch_size
        C = self.dim
        G = self.rt_per_window

        # TODO: Concatenate multi-scale tokens for each batch

        
        # DEBUG:
        pyramid_depths = list(relay_token_dict.keys())
        ct_temp = relay_token_dict[pyramid_depths[0]]
        ct_debug = self.ct_gamma1 * self.ct_attention(self.ct_norm1(ct_temp), octree, pyramid_depths[0])


        # TODO: Copy core code from CTAttention class. Need to concat CTs for
        #       all 3 scales in each batch, then pad, then need to fix the CT
        #       attn mask, then compute all the things and unpad + split CTs

        # ct = carrier_tokens
        
        # # Do global attention via carrier tokens
        # # NOTE: No longer using ADaPE per octformer block
        # # if self.use_ADaPE:
        # #     ct = ct + self.ct_adape(octree, depth)
        # ct_attn = self.ct_gamma1 * self.ct_attention(self.ct_norm1(ct), octree, depth)
        # ct = ct + self.ct_drop_path(ct_attn, octree, depth)
        # ct_ffn = self.ct_gamma2 * self.ct_mlp(self.ct_norm2(ct))
        # ct = ct + self.ct_drop_path(ct_ffn, octree, depth)
            
        return relay_token_dict


class RelayTokenInitialiser(torch.nn.Module):
    """
    Relay token Initialiser based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention.
    """
    
    def __init__(self, dim: int, patch_size: int, nempty: bool, conv_norm: str,
                 rt_size: int = 1, use_cpe: bool = False, xcpe: bool = False):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            patch_size: patch size.
            nempty: only compute on non-empty octree leaf nodes.
            conv_norm: type of normalisation to use after the conv layer.
            rt_size: number of relay tokens per local window.
            use_cpe: disable CPE during token initialisation.
            xcpe: Use xCPE instead of CPE.
        """
        super().__init__()
        self.use_cpe = use_cpe
        if use_cpe:
            self.cpe = CPE(dim, nempty=nempty, conv_norm=conv_norm, xcpe=xcpe)
        # NOTE: Currently, because of how octree windows are constructed,
        #       consecutive batch elements can have an octree window with
        #       elements from both batches. This means avgpooled features for
        #       'leaky' windows will contain features from 2 batch elements, and
        #       not be valid. The only way to prevent this (that I can tell) is
        #       to redo the OCNN batch implementation to include padding around
        #       each batch element. Instead, I opt to ignore 'leaky' window
        #       features during global attention. This should be fine most of
        #       the time as a max of 1 window will be ignored per batch element,
        #       typically out of 100s, but isn't the optimal solution.

        # Pool the features in each octree window, without considering surrounding features
        assert patch_size % rt_size == 0, "Currently, patch_size must be divisible by ct_size"
        # self.pool = torch.nn.AvgPool1d(kernel_size=patch_size//ct_size)
        self.patch_size = patch_size
        self.dim = dim
        self.rt_size = rt_size

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        K = self.patch_size
        C = self.dim
        G = self.rt_size
        
        if self.use_cpe:
            data = self.cpe(data, octree, depth)
        data = octree.patch_partition(data, depth)
        # Reshape to windows, and mask out ignored values as NaN
        # TODO: Make this work with rt_size > 1
        data = data.view(-1, K//G, C)
        mask = octree.ct_init_mask[depth].unsqueeze(-1)
        data = data.masked_fill(mask, value=torch.nan)
        # Avg pool over spatial dimension
        # NOTE: AvgPool1D can't handle NaNs, so use nanmean() instead
        relay_tokens = torch.nanmean(data, dim=1)
        assert(not torch.any(relay_tokens.isnan())), \
            "NaN propagated during relay token init, check code"
        return relay_tokens

class OctFormerStage(torch.nn.Module):

    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                 dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
                 disable_RPE: bool = False, use_ct: bool = False, ct_size: int = 1,
                 ct_propagation: bool = False,
                 ct_propagation_scale: Optional[float] = None,
                 ADaPE_mode: Optional[str] = None,
                 grad_checkpoint: bool = True, num_blocks: int = 2,
                 conv_norm: str = 'batchnorm', layer_scale: Optional[float] = None,
                 xcpe: bool = False, octformer_block=OctFormerBlock, **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        self.grad_checkpoint = grad_checkpoint
        self.use_ct = use_ct
        self.use_ADaPE = ADaPE_mode is not None
        # self.interval = interval  # normalisation interval
        # self.num_norms = (num_blocks - 1) // self.interval

        self.blocks = torch.nn.ModuleList([octformer_block(
            dim=dim, num_heads=num_heads, patch_size=patch_size,
            dilation=1 if (i % 2 == 0) else dilation,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            nempty=nempty, activation=activation, disable_RPE=disable_RPE,
            use_ct=use_ct, ct_size=ct_size, ct_propagation=ct_propagation,
            ct_propagation_scale=ct_propagation_scale, use_ADaPE=self.use_ADaPE,
            conv_norm=conv_norm, last=(i == num_blocks - 1),
            layer_scale=layer_scale, xcpe=xcpe) for i in range(num_blocks)])
        # self.norms = torch.nn.ModuleList([
        #     torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])
        if not self.use_ct:
            return
        self.global_tokeniser = RelayTokenInitialiser(dim, patch_size=patch_size,
                                                 nempty=nempty,
                                                 conv_norm=conv_norm,
                                                 rt_size=ct_size,
                                                 use_cpe=(not self.use_ADaPE),
                                                 xcpe=xcpe)
        if self.use_ADaPE:
            self.ct_adape = ADaPE(dim, activation, ADaPE_mode)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        ct = self.global_tokeniser(data, octree, depth) if self.use_ct else None
        # Inject positional encoding for CTs
        if self.use_ADaPE and ct is not None:
            ct = ct + self.ct_adape(octree, depth)
        for i in range(self.num_blocks):
            if self.grad_checkpoint and self.training:
                data, ct = checkpoint(self.blocks[i], data, ct, octree, depth, use_reentrant=False)  # disable reentrant to fix error with no_grad?
            else:
                data, ct = self.blocks[i](data, ct, octree, depth)
            # if i % self.interval == 0 and i != 0:
            #   data = self.norms[(i - 1) // self.interval](data)
        return data

class HOTFormerStage(torch.nn.Module):

    def __init__(self, dim: int, num_heads: int, num_blocks: int = 10,
                 num_pyramid_levels: int = 3, patch_size: int = 32,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU,
                 disable_RPE: bool = False, rt_size: int = 1,
                 ADaPE_mode: Optional[str] = None,
                 grad_checkpoint: bool = True, conv_norm: str = 'batchnorm',
                 layer_scale: Optional[float] = None, xcpe: bool = False, **kwargs):
        super().__init__()
        self.num_pyramid_levels = num_pyramid_levels
        self.num_blocks = num_blocks
        self.use_ADaPE = ADaPE_mode is not None
        self.grad_checkpoint = grad_checkpoint

        self.hosa_blocks = torch.nn.ModuleList([])
        for _ in range(self.num_pyramid_levels):
            self.hosa_blocks.append(torch.nn.ModuleList(
                [HOTFormerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    dilation=1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    nempty=nempty,
                    activation=activation,
                    disable_RPE=disable_RPE,
                    rt_size=rt_size,
                    use_ADaPE=self.use_ADaPE,
                    conv_norm=conv_norm,
                    layer_scale=layer_scale,
                    xcpe=xcpe
                ) for i in range(self.num_blocks)]
            ))
        self.rtsa_blocks = torch.nn.ModuleList(
            [RelayTokenTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                patch_size=patch_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                nempty=nempty,
                activation=activation,
                rt_size=rt_size,
                use_ADaPE=self.use_ADaPE,
                conv_norm=conv_norm,
                layer_scale=layer_scale,) for i in range(self.num_blocks)]
        )
        self.relay_tokeniser = RelayTokenInitialiser(
            dim=dim,
            patch_size=patch_size,
            nempty=nempty,
            conv_norm=conv_norm,
            rt_size=rt_size,
            use_cpe=(not self.use_ADaPE),
            xcpe=xcpe
        )
        self.downsamples = torch.nn.ModuleList(
            [Downsample(
                in_channels=dim,
                out_channels=dim,
                kernel_size=[2],
                nempty=nempty,
                conv_norm=conv_norm
            ) for _ in range(self.num_pyramid_levels - 1)]
        )
        if self.use_ADaPE:
            self.rt_adape = ADaPE(dim, activation, ADaPE_mode)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        # TODO: Take input local features, init features for d-1, d-2, init
        #       relay tokens for all scales
        pyramid_depths = [(depth - j) for j in range(self.num_pyramid_levels)]
        # Store local features and relay tokens by octree depth in dict
        local_feat_dict = {depth: data}
        relay_token_dict = {}

        # Initialise local features and relay tokens
        for j, depth_j in enumerate(pyramid_depths):
            relay_token_dict[depth_j] = self.relay_tokeniser(
                local_feat_dict[depth_j], octree, depth_j,
            )
            # TODO: Ensure ADaPE uses correct position coordinates,for
            #       multi-scale compatibility
            if self.use_ADaPE:  # Inject positional encoding for RTs
                relay_token_dict[depth_j] = (
                    relay_token_dict[depth_j] + self.rt_adape(octree, depth_j)
                )            
            if j < (self.num_pyramid_levels - 1):
                local_feat_dict[depth_j - 1] = self.downsamples[j](
                    local_feat_dict[depth_j], octree, depth_j,
                )

        # TODO: Begin loop of RTSA + H-OSA
        for i in range(self.num_blocks):
            # Compute global multi-scale interactions through RTSA
            if self.grad_checkpoint and self.training:
                relay_token_dict = checkpoint(
                    self.rtsa_blocks[i], relay_token_dict, octree,
                    use_reentrant=False,
                )
            else:
                relay_token_dict = self.rtsa_blocks[i](relay_token_dict, octree)
            
            # Propagate to local features with H-OSA
            for j, depth_j in enumerate(pyramid_depths):
                if self.grad_checkpoint and self.training:
                    local_feat_dict[depth_j], relay_token_dict[depth_j] = \
                        checkpoint(
                            self.hosa_blocks[j][i], local_feat_dict[depth_j],
                            relay_token_dict[depth_j], octree, depth_j,
                            use_reentrant=False,
                        )
                else:
                    local_feat_dict[depth_j], relay_token_dict[depth_j] = \
                        self.hosa_blocks[j][i](
                            local_feat_dict[depth_j], relay_token_dict[depth_j],
                            octree, depth_j,
                        )
            
        # TODO: Work out how batching will work with multi-scale RTs?
                
            
        # if self.grad_checkpoint and self.training:
        #     data, ct = checkpoint(self.blocks[i], data, ct, octree, depth, use_reentrant=False)
        # else:
        #     data, ct = self.blocks[i](data, ct, octree, depth)

        # TODO: Return local feats + RTs for all scales

        
        
        return local_feat_dict, relay_token_dict
        


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


class HOTFormerBase(torch.nn.Module):

    def __init__(self, in_channels: int,
                 channels: List[int] = [128, 256],
                 num_blocks: List[int] = [4, 10],
                 num_heads: Optional[List[int]] = [8, 16],
                 num_pyramid_levels: int = 3,
                 patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,
                 nempty: bool = True, stem_down: int = 2, rt_size: int = 1,
                 ADaPE_mode: Optional[str] = None,
                 grad_checkpoint: bool = True,
                 downsample_input_embeddings: bool = True,
                 disable_RPE: bool = False, conv_norm: str = 'batchnorm',
                 layer_scale: Optional[float] = None, xcpe: bool = False,
                 **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.dilation = dilation
        self.nempty = nempty
        self.num_stages = 1 + num_pyramid_levels
        self.num_pyramid_levels = num_pyramid_levels
        self.stem_down = stem_down
        self.downsample_input_embeddings = downsample_input_embeddings
        # if len(ct_layers) < len(channels):
        #     ct_layers += (False,) * (len(channels)-len(ct_layers))
        # self.ct_layers = ct_layers
        # ct_size = ct_size if any(ct_layers) else 0
        self.ct_size = rt_size
        self.ADaPE_mode = ADaPE_mode
        self.use_ADaPE = (ADaPE_mode is not None)
        # Stochastic depth per block
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down,
                                      nempty, downsample_input_embeddings,
                                      conv_norm)
        if num_heads is None:
            num_heads = [channel // 16 for channel in channels]

        self.octf_stage = OctFormerStage(
            dim=channels[0], num_heads=num_heads[0], patch_size=patch_size,
            drop_path=drop_ratio[sum(num_blocks[:0]):sum(num_blocks[:0+1])],
            dilation=dilation, nempty=nempty, disable_RPE=disable_RPE,
            grad_checkpoint=grad_checkpoint, num_blocks=num_blocks[0],
            conv_norm=conv_norm, layer_scale=layer_scale, xcpe=xcpe)
        self.downsample = Downsample(
            channels[0], channels[1], kernel_size=[2], nempty=nempty,
            conv_norm=conv_norm)
        self.hotf_stage = HOTFormerStage(
            dim=channels[1], num_heads=num_heads[1], num_blocks=num_blocks[1],
            num_pyramid_levels=num_pyramid_levels, patch_size=patch_size,
            drop_path=drop_ratio[sum(num_blocks[:1]):sum(num_blocks[:1+1])],
            nempty=nempty, disable_RPE=disable_RPE,
            grad_checkpoint=grad_checkpoint, conv_norm=conv_norm,
            rt_size=rt_size, ADaPE_mode=ADaPE_mode, layer_scale=layer_scale,
            xcpe=xcpe)
        
        # self.oct_layers = torch.nn.ModuleList([OctFormerStage(
        #         dim=channels[i], num_heads=num_heads[i], patch_size=patch_size,
        #         drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i+1])],
        #         dilation=dilation, nempty=nempty, disable_RPE=disable_RPE,
        #         grad_checkpoint=grad_checkpoint, num_blocks=num_blocks[i],
        #         conv_norm=conv_norm, use_ct=ct_layers[i], ct_size=ct_size,
        #         ct_propagation=ct_propagation,
        #         ct_propagation_scale=ct_propagation_scale,
        #         ADaPE_mode=ADaPE_mode, layer_scale=layer_scale,
        #         xcpe=xcpe) for i in range(self.num_stages)])
        # self.downsamples = torch.nn.ModuleList([Downsample(
        #         channels[i], channels[i + 1], kernel_size=[2], nempty=nempty,
        #         conv_norm=conv_norm) for i in range(self.num_stages - 1)])
        # self.hot_layers = None

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        # Generate initial convolution embeddings
        data = self.patch_embed(data, octree, depth)
        if self.downsample_input_embeddings:
            depth = depth - self.stem_down   # current octree depth
        octree = OctreeT(octree, self.patch_size, self.dilation, self.nempty,
                         max_depth=depth, start_depth=depth-self.num_stages+1,
                         ct_layers=[False]+[True]*self.num_pyramid_levels,
                         ct_size=self.ct_size, ADaPE_mode=self.ADaPE_mode)
        octree.build_t()
        
        # Refine local features with standard octree attention
        features = {}
        data = self.octf_stage(data, octree, depth)
        features[depth] = data
        data = self.downsample(data, octree, depth)
        depth = depth - 1
        # TODO: Get correct output from HOTF parallel stages
        data = self.hotf_stage(data, octree, depth)

        # # TODO: Begin hierarchical attention
        # for i in range(1, self.num_stages):
        #     depth_i = depth - i
        #     data = self.layers[i](data, octree, depth_i)
        #     features[depth_i] = data
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


class HOTFormer(torch.nn.Module):
    """
    HOTFormer backbone class adapted from https://github.com/octree-nn/octformer,
    with Hierarchical Attention (HAT) design inspired by
    https://github.com/NVlabs/FasterViT.
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int] = [96, 192, 384, 384],
        num_blocks: List[int] = [2, 2, 6, 2],
        num_heads: Optional[List[int]] = [6, 12, 24, 24],
        num_pyramid_levels: int = 3,
        patch_size: int = 32,
        dilation: int = 4,
        drop_path: float = 0.5,
        nempty: bool = True,
        stem_down: int = 2,
        num_top_down: int = 2,
        fpn_channel: int = 168,
        rt_size: int = 1,
        ADaPE_mode: Optional[str] = None,
        grad_checkpoint: bool = True,
        downsample_input_embeddings: bool = True,
        disable_RPE: bool = False,
        conv_norm: str = 'batchnorm',
        layer_scale: Optional[float] = None,
        qkv_init: List = ['trunc_normal', 0.02],
        xcpe: bool = False,
        **kwargs
    ):
        """
        Args:
            in_channels: Number of input channels, typically 3 if only using x,y,z information.
            channels: List containing number of feature channels per stage.
            num_blocks: List containing number of OctFormer blocks per stage.
            num_heads: List containing number of attention heads per stage, defaults to channel_size//16.
            num_pyramid_levels: Number of octree levels to consider for hierarchical attention.
            patch_size: Size of local attention patches/windows, constructed using z-order curve traversal.
            dilation: Dilation amount for Octree attention
            drop_path: Stochastic depth probability (this is the max value stochastic depth scales to).
            nempty: Boolean indicating if only non-empty octants should be used (set True for sparse operation).
            rt_size: Size of carrier tokens, note that patch_size must be divisible by this.
            ADaPE_mode: Use Absolute Distribution-aware Position Encoding (ADaPE) during carrier token attention. Mode (valid values: ['pos','var','cov']) determines whether position, variance, or covariance is used (cumulative aggregation of those three)
            num_top_down: Number of top-down layers in FPN. Output features will be at Octree depth d = octree_depth - stem_down - (num_stages - 1) + num_top_down.
            fpn_channel: Number of channels in FPN top-down branch, default is to set this equal to number of channels used in Pooling (i.e. output_dim param).
            grad_checkpoint: Use gradient checkpoint to save memory, at cost of extra computation time.
            downsample_input_embeddings: Do downsampling in input conv embedding.
            disable_RPE: Disable RPE during self-attention (only applies to local attention).
            conv_norm: Type of normalisation used after convolution layers, valid params are in ['batchnorm', 'layernorm', 'powernorm'].
            layer_scale: Coefficient to initialise learnable channel-wise scale multipliers for attention outputs, or None to disable this.
            qkv_init: Method of initialisation to use for qkv linear layers
            xcpe: Use xCPE instead of CPE (from PointTransformerV3)
        """
        super().__init__()
        self.backbone = HOTFormerBase(
            in_channels=in_channels,
            channels=channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_pyramid_levels=num_pyramid_levels,
            patch_size=patch_size,
            dilation=dilation,
            drop_path=drop_path,
            nempty=nempty,
            stem_down=stem_down,
            rt_size=rt_size,
            ADaPE_mode=ADaPE_mode,
            grad_checkpoint=grad_checkpoint,
            downsample_input_embeddings=downsample_input_embeddings,
            disable_RPE=disable_RPE,
            conv_norm=conv_norm,
            layer_scale=layer_scale,
            xcpe=xcpe,
        )
        # TODO: replace or remove FPN header
        self.head = FPNHeader(
            channels=channels,
            fpn_channel=fpn_channel,
            nempty=nempty,
            num_top_down=num_top_down,
            conv_norm=conv_norm,
        )
        self.qkv_init = qkv_init
        self.apply(self.init_weights)
        # Apply special initialisation to qkv linear layers
        for m in self.named_modules():
            if 'qkv' in m[0]:
                self.init_qkv_weights(m[1])

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)        
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def init_qkv_weights(self, m):
        if not isinstance(m, torch.nn.Linear):
            return

        if self.qkv_init[0] == 'torch_default':
            return
        elif self.qkv_init[0] == 'trunc_normal':
            torch.nn.init.trunc_normal_(m.weight, std=self.qkv_init[1])
        elif self.qkv_init[0] == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        elif self.qkv_init[0] == 'xavier_normal':
            torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        elif self.qkv_init[0] == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif self.qkv_init[0] == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
            raise ValueError("Invalid init type")

        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        features = self.backbone(data, octree, depth)
        # TODO: replace or remove FPN header
        output, output_depth = self.head(features, octree)
        return output, output_depth
