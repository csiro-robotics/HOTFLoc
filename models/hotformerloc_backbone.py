# --------------------------------------------------------
# HOTFormerLoc: Hierarchical Octree Transformer for
# Ground-Aerial Lidar Place Recognition in Natural
# Environments
#
# Adapted from https://github.com/octree-nn/octformer by
# Ethan Griffiths (Data61, Pullenvale)
# --------------------------------------------------------
import torch
from torch import Tensor
import torch.nn.functional as F

from ocnn.octree import Octree
from typing import Optional, List, Dict
from torch.utils.checkpoint import checkpoint
from models.octree import OctreeT
from models.layers.octformer_layers import (
    MLP, CPE, ADaPE, OctreeDropPath
)
from models.octformer_backbone import (
    PatchEmbed, Downsample, OctreeAttention, OctFormerStage, OctFormerBlock
)
from models.relay_token_utils import concat_and_pad_rt, unpad_and_split_rt


class RTAttention(torch.nn.Module):
    """
    Attention block for relay token self attention (RTSA). Assumes multi-scale
    relay tokens have already been combined in batches with padding.
    """
    def __init__(self, dim: int, patch_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 use_rpe: bool = True, return_attn_maps: bool = False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rpe = use_rpe  # TODO: implement RPE
        self.return_attn_maps = return_attn_maps
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)

        # NOTE: RPE table is currently constructed using relative pos of
        #       octree nodes, but this is not so easy to do for RTs. Need to
        #       find another solution.
        # self.rpe = RPE(patch_size, num_heads) if use_rpe else None

    def forward(self, relay_tokens: Tensor, octree: OctreeT):
        attn_dict = {}
        B = octree.batch_size
        H = self.num_heads
        C = self.dim
        rt = relay_tokens

        # get rt attn mask
        attn_mask = octree.rt_attn_mask

        # qkv
        qkv = self.qkv(rt)
        # log qkv std loss over batch dim (hinge loss on std)
        qkv_std = qkv.view(-1, C*3).std(dim=0)
        # qkv_std_loss = torch.mean(F.relu(1 - qkv_std))

        qkv = qkv.reshape(B, -1, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, K, C')

        if (
            torch.__version__ >= torch.torch_version.TorchVersion(2.0)
            and not self.return_attn_maps
        ):
            #### EFFICIENT IMPLEMENTATION ####
            attn_mask = attn_mask.to(q.dtype).unsqueeze(1)
            rt = F.scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=attn_mask,
            ).transpose(1, 2).reshape(B, -1, C)  # (B, K, C)
            ####
        else:
            #### ORIGINAL IMPLEMENTATION ####
            q_temp = q * self.scale
            # attn
            attn_map = q_temp @ k.transpose(-2, -1)        # (B, H, K, K)
            # attn = self.apply_rpe(attn, rel_pos)  # TODO: implement RPE
            attn_map = attn_map + attn_mask.unsqueeze(1)
            attn = self.softmax(attn_map)
            attn = self.attn_drop(attn)
            rt = (attn @ v).transpose(1, 2).reshape(B, -1, C)  # (B, K, C)
            ####
        
        if self.return_attn_maps:
            attn_dict = {'attn_map': attn_map, 'q': q_temp, 'k': k, 'v': v}
        attn_dict['qkv_std'] = qkv_std

        # ffn
        rt = self.proj(rt)
        rt = self.proj_drop(rt)
        return rt, attn_dict
    
    def apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            attn = attn + self.rpe(rel_pos)
        return attn

    def extra_repr(self) -> str:
        return 'dim={}, num_heads={}'.format(self.dim, self.num_heads)


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
                 rt_rpe_init: bool = False, 
                 disable_RPE: bool = False, conv_norm: str = 'batchnorm',
                 last: bool = False, layer_scale: Optional[float] = None,
                 xcpe: bool = False,
                 return_feats_and_attn_maps: bool = False, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.return_feats_and_attn_maps = return_feats_and_attn_maps
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
        self.attention = OctreeAttention(
            dim, patch_size, num_heads, qkv_bias, qk_scale, attn_drop,
            proj_drop, dilation, ct_per_window=rt_per_window,
            use_rpe=(not disable_RPE), ct_rpe_init=rt_rpe_init,
            return_attn_maps=return_feats_and_attn_maps,
        )
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

    def forward(self, data: Tensor, relay_tokens: Tensor, octree: OctreeT, depth: int):
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
        attn, attn_dict = self.attention(self.norm1(data), octree, depth)
        attn = self.gamma1 * attn
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
            # Mask out padded and overlap RTs
            rt_upsampled = rt_upsampled.masked_fill(mask, value=0).view(-1, C)
            rt_upsampled = octree.patch_reverse(rt_upsampled, depth)
            data = data + self.rt_gamma_propagate*rt_upsampled
        return data, rt, attn_dict


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
                 layer_scale: Optional[float] = None,
                 return_feats_and_attn_maps: bool = False, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        # self.use_ADaPE = use_ADaPE,
        self.dim = dim
        self.return_feats_and_attn_maps = return_feats_and_attn_maps
        rt_per_window = rt_size  # track number of carrier tokens per window
        self.rt_per_window = rt_per_window
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]

        # NOTE: No longer using ADaPE per octformer block
        # if self.use_ADaPE:
        #     self.rt_adape = ADaPE(dim, activation)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.rt_attention = RTAttention(dim, patch_size, num_heads, qkv_bias,
                                        qk_scale, attn_drop, proj_drop,
                                        return_attn_maps=return_feats_and_attn_maps)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.drop_path = OctreeDropPath(drop_path, nempty, use_ct=True)
        self.gamma1 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma2 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, relay_token_dict: Dict[int, Tensor], octree: OctreeT,
                rt_cls_token: Optional[Tensor] = None):
        pyramid_depths = octree.pyramid_depths

        # # TODO: TIME THIS FUNC, CHECK IF EFFICIENCY IMPROVEMENTS NEEDED (seems okay, ~2.4ms with bs 128)
        # Concatenate and pad multi-scale relay tokens per batch
        rt = concat_and_pad_rt(relay_token_dict, octree, pyramid_depths)
        B, N, C = rt.shape
       
        if rt_cls_token is not None:
            rt = torch.cat((rt_cls_token, rt), dim=1)

        # Generate drop_path batch idx (simple for padded tokens, each row is a batch)
        drop_path_batch_idx = self.get_drop_path_idx(rt)

        # Compute global attention via relay tokens
        # NOTE: No longer using ADaPE per octformer block
        # if self.use_ADaPE:
        #     rt = rt + self.rt_adape(octree, depth)
        rt_attn, rt_attn_dict = self.rt_attention(self.norm1(rt), octree)
        rt_attn = self.gamma1 * rt_attn
        rt = rt + self.drop_path(rt_attn, octree, batch_id=drop_path_batch_idx)
        rt_ffn = self.gamma2 * self.mlp(self.norm2(rt))
        rt = rt + self.drop_path(rt_ffn, octree, batch_id=drop_path_batch_idx)

        # Split cls token
        if rt_cls_token is not None:
            rt_cls_token, rt = rt.split([1, N], dim=1)
        
        # # TODO: TIME THIS FUNC, CHECK IF EFFICIENCY IMPROVEMENTS NEEDED (seems okay, ~3.5ms with bs 128)
        # Unpad + split CTs
        relay_token_dict = unpad_and_split_rt(rt, octree, pyramid_depths)
        return relay_token_dict, rt_attn_dict, rt_cls_token

    def get_drop_path_idx(self, relay_tokens: Tensor) -> Tensor:
        """Compute the batch idx tensor for drop_path."""
        B, N = relay_tokens.shape[:2]
        drop_path_batch_idx = torch.zeros((B, N), dtype=torch.long)
        drop_path_batch_idx += torch.arange(B).unsqueeze(1)
        return drop_path_batch_idx
        

class RelayTokenInitialiser(torch.nn.Module):
    """
    Relay token Initialiser based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention.
    """
    
    def __init__(self, dim: int, patch_size: int, nempty: bool, conv_norm: str,
                 rt_size: int = 1, use_cpe: bool = False, xcpe: bool = False,
                 rt_init_type: str = 'avg_pool'):
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
            init_type: Type of initialisation to use. Valid values are ('avg_pool', 'max_pool', 'learnable').
        """
        super().__init__()
        VALID_INIT_TYPES = ('avg_pool', 'max_pool', 'learnable')
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
        rt_init_type = rt_init_type.lower()
        if rt_init_type not in VALID_INIT_TYPES:
            raise ValueError()
        self.rt_init_type = rt_init_type
        self.rt_init_token = None
        if rt_init_type == 'learnable':
            self.rt_init_token = torch.nn.Parameter(torch.zeros(1, self.dim))
            torch.nn.init.normal_(self.rt_init_token, std=1e-6)  # based on timm ViT cls token init

    def forward(self, data: Tensor, octree: OctreeT, depth: int):
        K = self.patch_size
        C = self.dim
        G = self.rt_size
 
        if self.use_cpe:  # CPE disabled when using ADaPE
            data = self.cpe(data, octree, depth)
        data = octree.patch_partition(data, depth)
        # Reshape to windows
        # TODO: Make this work with rt_size > 1
        data = data.view(-1, K//G, C)
        mask = octree.ct_init_mask[depth].unsqueeze(-1)
        if self.rt_init_type == 'avg_pool':
            # Mask out ignored values with NaNs
            data = data.masked_fill(mask, value=torch.nan)
            # Avg pool over spatial dimension
            # NOTE: AvgPool1D can't handle NaNs, so use nanmean() instead
            relay_tokens = torch.nanmean(data, dim=1)
            assert(not torch.any(relay_tokens.isnan())), \
                "NaN propagated during relay token init, check code"
        elif self.rt_init_type == 'max_pool':
            # Mask out ignored values with sufficiently negative values
            data = data.masked_fill(mask, value=-torch.inf)
            # Max pool over spatial dimension
            relay_tokens = torch.max(data, dim=1).values
        elif self.rt_init_type == 'learnable':
            # TODO: Likely need to add something here to prevent RT uniformity 
            #       (ADaPE should contribute somewhat this, or could swap order
            #       of RTSA and H-OSA so that RTs aggregate local window feats)
            N = data.size(0)
            relay_tokens = self.rt_init_token.expand(N, C).clone()
        else:
            raise ValueError()
        return relay_tokens
    

class HOTFormerStage(torch.nn.Module):

    def __init__(self, channels: List[int], num_heads: List[int],
                 num_blocks: int = 10, num_pyramid_levels: int = 3,
                 patch_size: int = 32, dilation: int = 1, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU,
                 disable_RPE: bool = False, rt_size: int = 1,
                 rt_propagation: bool = False,
                 rt_propagation_scale: Optional[float] = None,
                 rt_init_type: str = 'avg_pool',
                 rt_rpe_init: bool = False, 
                 rt_class_token: bool = False,
                 disable_rt: bool = False,
                 ADaPE_mode: Optional[str] = None,
                 grad_checkpoint: bool = True, conv_norm: str = 'batchnorm',
                 layer_scale: Optional[float] = None, xcpe: bool = False,
                 return_feats_and_attn_maps: bool = False, **kwargs):
        super().__init__()
        self.use_projections = True
        # Handle case where a single num channels or heads is specified for all levels
        if len(channels) == 1:
            channels = channels * num_pyramid_levels
            self.use_projections = False
        if len(num_heads) == 1:
            num_heads = num_heads * num_pyramid_levels
        assert len(channels) == num_pyramid_levels, "Invalid num channels specified"
        assert len(num_heads) == num_pyramid_levels, "Invalid num heads specified"

        self.channels = channels
        self.num_heads = num_heads
        self.disable_rt = disable_rt
        if self.disable_rt:
            self.use_projections = False
        self.max_rt_channels = max(channels)
        self.max_rt_num_heads = num_heads[channels.index(self.max_rt_channels)]
        self.num_pyramid_levels = num_pyramid_levels
        self.num_blocks = num_blocks
        self.use_ADaPE = ADaPE_mode is not None
        self.grad_checkpoint = grad_checkpoint
        self.return_feats_and_attn_maps = return_feats_and_attn_maps

        # Class token
        self.rt_cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.max_rt_channels)) if rt_class_token else None
        if self.rt_cls_token is not None:
            torch.nn.init.normal_(self.rt_cls_token, std=1e-6)

        self.hosa_blocks = torch.nn.ModuleList([])
        if self.use_projections:
            self.up_projections = torch.nn.ModuleList([])
            self.down_projections = torch.nn.ModuleList([])
            self.init_up_projections = torch.nn.ModuleList([])
        
        for j in range(self.num_pyramid_levels):
            hosa_blocks_j = torch.nn.ModuleList([])
            up_projections_j = torch.nn.ModuleList([])
            down_projections_j = torch.nn.ModuleList([])
            for i in range(self.num_blocks):
                if not self.disable_rt:  # for ablations disabling RT attn
                    hosa_blocks_j.append(HOTFormerBlock(
                        dim=channels[j],
                        num_heads=num_heads[j],
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
                        rt_propagation=rt_propagation,
                        rt_propagation_scale=rt_propagation_scale,
                        rt_rpe_init=rt_rpe_init, 
                        use_ADaPE=self.use_ADaPE,
                        conv_norm=conv_norm,
                        last=(i == self.num_blocks - 1),
                        layer_scale=layer_scale,
                        xcpe=xcpe,
                        return_feats_and_attn_maps=return_feats_and_attn_maps,
                    ))
                else:
                    hosa_blocks_j.append(OctFormerBlock(
                        dim=channels[j],
                        num_heads=num_heads[j],
                        patch_size=patch_size,
                        dilation=1 if (i % 2 == 0) else dilation,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        nempty=nempty,
                        activation=activation,
                        disable_RPE=disable_RPE,
                        conv_norm=conv_norm,
                        last=(i == self.num_blocks - 1),
                        layer_scale=layer_scale,
                        xcpe=xcpe,
                        return_feats_and_attn_maps=return_feats_and_attn_maps,
                    ))                    
                if self.use_projections:
                    up_projections_j.append(torch.nn.Linear(
                        in_features=channels[j],
                        out_features=self.max_rt_channels,
                    ))
                    down_projections_j.append(torch.nn.Linear(
                        in_features=self.max_rt_channels,
                        out_features=channels[j],
                    ))
            self.hosa_blocks.append(hosa_blocks_j)
            if self.use_projections:
                self.up_projections.append(up_projections_j)
                self.down_projections.append(down_projections_j)
                self.init_up_projections.append(torch.nn.Linear(
                    in_features=channels[j],
                    out_features=self.max_rt_channels,
                ))
        if not self.disable_rt:
            self.rtsa_blocks = torch.nn.ModuleList(
                [RelayTokenTransformerBlock(  # Largest channel size for output
                    dim=self.max_rt_channels,
                    num_heads=self.max_rt_num_heads,
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
                    layer_scale=layer_scale,
                    return_feats_and_attn_maps=return_feats_and_attn_maps,
                ) for i in range(self.num_blocks)]
            )
            if self.use_projections:
                self.relay_tokeniser = torch.nn.ModuleList(
                    [RelayTokenInitialiser(
                        dim=channels[j],
                        patch_size=patch_size,
                        nempty=nempty,
                        conv_norm=conv_norm,
                        rt_size=rt_size,
                        use_cpe=(not self.use_ADaPE),
                        xcpe=xcpe,
                        rt_init_type=rt_init_type,
                    ) for j in range(self.num_pyramid_levels)]
                )
            else:
                self.relay_tokeniser = RelayTokenInitialiser(
                    dim=self.max_rt_channels,
                    patch_size=patch_size,
                    nempty=nempty,
                    conv_norm=conv_norm,
                    rt_size=rt_size,
                    use_cpe=(not self.use_ADaPE),
                    xcpe=xcpe,
                )
            if self.use_ADaPE:
                self.rt_adape = ADaPE(self.max_rt_channels, activation, ADaPE_mode)
                if self.use_projections:
                    self.rt_adape_projections = torch.nn.ModuleList(
                        [torch.nn.Linear(
                            in_features=self.max_rt_channels,
                            out_features=self.channels[j],
                        ) for j in range(self.num_pyramid_levels)]
                    )

        self.downsamples = torch.nn.ModuleList(
            [Downsample(
                in_channels=channels[j],
                out_channels=channels[j+1],
                kernel_size=[2],
                nempty=nempty,
                conv_norm=conv_norm,
            ) for j in range(self.num_pyramid_levels - 1)]
        )

    def init_pyramid_feats(self, data: Tensor, octree: OctreeT):
        # Store local features and relay tokens by octree depth in dict
        local_feat_dict = {self.pyramid_depths[0]: data}
        relay_token_dict = {}

        # Initialise local features and relay tokens
        for j, depth_j in enumerate(self.pyramid_depths):
            if not self.disable_rt:
                if self.use_projections:
                    relay_token_dict[depth_j] = self.relay_tokeniser[j](
                        local_feat_dict[depth_j], octree, depth_j,
                    )
                    if self.use_ADaPE:  # Inject positional encoding for RTs
                        relay_token_dict[depth_j] = (
                            relay_token_dict[depth_j]
                            + self.rt_adape_projections[j](self.rt_adape(octree, depth_j))
                        )
                else:
                    relay_token_dict[depth_j] = self.relay_tokeniser(
                        local_feat_dict[depth_j], octree, depth_j,
                    )
                    if self.use_ADaPE:  # Inject positional encoding for RTs
                        relay_token_dict[depth_j] = (
                            relay_token_dict[depth_j] + self.rt_adape(octree, depth_j)
                        )
            else:
                relay_token_dict[depth_j] = None

            if j < (self.num_pyramid_levels - 1):
                local_feat_dict[depth_j - 1] = self.downsamples[j](
                    local_feat_dict[depth_j], octree, depth_j,
                )
        return local_feat_dict, relay_token_dict
    
    def forward(self, data: Tensor, octree: OctreeT, depth: int) -> tuple[dict, dict, list]:
        rt_cls_token = self.rt_cls_token
        if rt_cls_token is not None:
            rt_cls_token = rt_cls_token.expand(octree.batch_size, -1, -1)
        self.pyramid_depths = octree.pyramid_depths
        feats_and_attn_maps_per_block = [{'local_qkv_std': {}} for _ in range(self.num_blocks)]
        if self.return_feats_and_attn_maps:
            [feats_and_attn_maps_per_block[i].update(
                {'local_attn': {}, 'local_feats': {}, 'rt_feats_post_local': {}}
            ) for i in range(self.num_blocks)]
        # Initialise local features + relay token dicts
        local_feat_dict, relay_token_dict = self.init_pyramid_feats(data,
                                                                    octree)
        if self.use_projections and not self.disable_rt:  # Project RTs to same channel size
            for j, depth_j in enumerate(self.pyramid_depths):
                # Project RTs to global channel dim
                relay_token_dict[depth_j] = self.init_up_projections[j](
                    relay_token_dict[depth_j]
                )
        
        # Begin loop of RTSA + H-OSA
        for i in range(self.num_blocks):
            # Compute global multi-scale interactions through RTSA
            if not self.disable_rt:
                if self.grad_checkpoint and self.training:
                    relay_token_dict, relay_token_attn_dict, rt_cls_token = checkpoint(
                        self.rtsa_blocks[i], relay_token_dict, octree,
                        rt_cls_token, use_reentrant=False,
                    )
                else:
                    relay_token_dict, relay_token_attn_dict, rt_cls_token = self.rtsa_blocks[i](
                        relay_token_dict, octree, rt_cls_token,
                    )

                # Log qkv std (removed from dict so gradients are not detached in next step)
                feats_and_attn_maps_per_block[i].update({
                    'rt_qkv_std': relay_token_attn_dict.pop('qkv_std')
                })

                # Log rt cls token
                if rt_cls_token is not None:
                    # Log final attn vals without detaching gradients
                    feats_and_attn_maps_per_block[i].update({
                        'rt_cls_token': rt_cls_token,
                    })
                    if i == (self.num_blocks - 1):
                        feats_and_attn_maps_per_block[i].update({
                            'rt_final_cls_attn_vals': 
                                # how much cls token attends to RTs, averaged over heads (note this contains padding)
                                F.softmax(relay_token_attn_dict['attn_map'], dim=-1).mean(1)[:, 0, 1:],
                        })
                
                # Log feats and attn maps
                if self.return_feats_and_attn_maps:
                    feats_and_attn_maps_per_block[i].update({
                        'rt_attn': {
                            k: v.detach().cpu()
                            for k, v in relay_token_attn_dict.items()
                        },
                        'rt_feats_pre_local': {
                            k: v.detach().cpu()
                            for k, v in relay_token_dict.items()
                        },
                    })
            
            for j, depth_j in enumerate(self.pyramid_depths):
                # Project RTs back to local feat channel size
                if self.use_projections and not self.disable_rt:
                    relay_token_dict[depth_j] = self.down_projections[j][i](
                        relay_token_dict[depth_j]
                    )
                # Propagate to local features with H-OSA
                if self.grad_checkpoint and self.training:
                    (local_feat_dict[depth_j], relay_token_dict[depth_j],
                     local_attn_dict_depth_j) = (
                        checkpoint(
                            self.hosa_blocks[j][i], local_feat_dict[depth_j],
                            relay_token_dict[depth_j], octree, depth_j,
                            use_reentrant=False,
                        )
                    )
                else:
                    (local_feat_dict[depth_j], relay_token_dict[depth_j],
                     local_attn_dict_depth_j) = (
                        self.hosa_blocks[j][i](
                            local_feat_dict[depth_j], relay_token_dict[depth_j],
                            octree, depth_j,
                        )
                    )

                # Log qkv std (removed from dict so gradients are not detached in next step)
                if 'qkv_std' in local_attn_dict_depth_j:
                    qkv_std_temp = local_attn_dict_depth_j.pop('qkv_std')
                elif 'local_qkv_std' in local_attn_dict_depth_j:
                    qkv_std_temp = local_attn_dict_depth_j.pop('local_qkv_std')
                else:
                    raise KeyError
                feats_and_attn_maps_per_block[i]['local_qkv_std'].update({
                    depth_j: qkv_std_temp
                })
                
                # Log feats and attn maps
                if self.return_feats_and_attn_maps:
                    if not self.disable_rt:
                        temp_attn_dict = {
                            depth_j: {k: v.detach().cpu()
                                      for k, v in local_attn_dict_depth_j.items()},
                        }
                    else:  # Octformer blocks have different dict layout, so need to get 'local_attn' key here
                        temp_attn_dict = {
                            depth_j: {k: v.detach().cpu()
                                      for k, v in local_attn_dict_depth_j['local_attn'].items()},
                        }
                    feats_and_attn_maps_per_block[i]['local_attn'].update(temp_attn_dict)
                    feats_and_attn_maps_per_block[i]['local_feats'].update({
                        depth_j: local_feat_dict[depth_j].detach().cpu(),
                    })
                    if not self.disable_rt:
                        feats_and_attn_maps_per_block[i]['rt_feats_post_local'].update({
                            depth_j: relay_token_dict[depth_j].detach().cpu(),
                        })
                    
                # Project RTs to global channel dim
                if self.use_projections and not self.disable_rt:
                    relay_token_dict[depth_j] = self.up_projections[j][i](
                        relay_token_dict[depth_j]
                    )

        return local_feat_dict, relay_token_dict, feats_and_attn_maps_per_block


class HOTFormerBase(torch.nn.Module):

    def __init__(self, in_channels: int,
                 channels: List[int] = [128, 256],
                 num_blocks: List[int] = [4, 10],
                 num_heads: Optional[List[int]] = [8, 16],
                 num_pyramid_levels: int = 3, num_octf_levels: int = 1,
                 patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,
                 nempty: bool = True, stem_down: int = 2, rt_size: int = 1,
                 rt_propagation: bool = False,
                 rt_propagation_scale: Optional[float] = None,
                 rt_init_type: str = 'avg_pool',
                 rt_rpe_init: bool = False, 
                 rt_class_token: bool = False,
                 disable_rt: bool = False,
                 ADaPE_mode: Optional[str] = None,
                 ADaPE_use_accurate_point_stats: bool = False,
                 grad_checkpoint: bool = True,
                 downsample_input_embeddings: bool = True,
                 disable_RPE: bool = False, conv_norm: str = 'batchnorm',
                 layer_scale: Optional[float] = None, xcpe: bool = False,
                 return_feats_and_attn_maps: bool = False, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.dilation = dilation
        self.nempty = nempty
        self.num_pyramid_levels = num_pyramid_levels
        self.num_octf_levels = num_octf_levels
        self.num_stages = num_octf_levels + num_pyramid_levels
        self.stem_down = stem_down
        self.downsample_input_embeddings = downsample_input_embeddings
        self.ct_size = rt_size
        self.rt_class_token = rt_class_token
        self.ADaPE_mode = ADaPE_mode
        self.use_ADaPE = (ADaPE_mode is not None)
        self.ADaPE_use_accurate_point_stats = ADaPE_use_accurate_point_stats
        self.disable_rt = disable_rt
        self.return_feats_and_attn_maps = return_feats_and_attn_maps
        # Stochastic depth per block
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down,
                                      nempty, downsample_input_embeddings,
                                      conv_norm)
        if num_heads is None:
            num_heads = [channel // 16 for channel in channels]

        self.octf_stage = torch.nn.ModuleList([OctFormerStage(
            dim=channels[i], num_heads=num_heads[i], patch_size=patch_size,
            drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i+1])],
            dilation=dilation, nempty=nempty, disable_RPE=disable_RPE,
            use_ct=False, grad_checkpoint=grad_checkpoint,
            num_blocks=num_blocks[i], conv_norm=conv_norm,
            layer_scale=layer_scale, xcpe=xcpe) for i in range(num_octf_levels)])
        self.downsample = torch.nn.ModuleList([Downsample(
            channels[i], channels[i+1], kernel_size=[2], nempty=nempty,
            conv_norm=conv_norm) for i in range(num_octf_levels)])
        # Use last element of num_blocks for all HOTFormer levels (same for
        #   channels and num_heads, but can also specify per level)
        self.hotf_stage = HOTFormerStage(
            channels=channels[num_octf_levels:],
            num_heads=num_heads[num_octf_levels:], num_blocks=num_blocks[-1],
            num_pyramid_levels=num_pyramid_levels, patch_size=patch_size,
            dilation=dilation,
            drop_path=drop_ratio[sum(num_blocks[:-1]):sum(num_blocks[:])],
            nempty=nempty, disable_RPE=disable_RPE,
            grad_checkpoint=grad_checkpoint, conv_norm=conv_norm,
            rt_size=rt_size, rt_propagation=rt_propagation,
            rt_propagation_scale=rt_propagation_scale, rt_init_type=rt_init_type,
            rt_rpe_init=rt_rpe_init, rt_class_token=rt_class_token,
            disable_rt=disable_rt, ADaPE_mode=ADaPE_mode,
            layer_scale=layer_scale, xcpe=xcpe,
            return_feats_and_attn_maps=return_feats_and_attn_maps)

    def forward(self, data: Tensor, batch: Dict, depth: int):
        octree = batch['octree']
        points = batch['points'] if 'points' in batch else None
        # Generate initial convolution embeddings
        data = self.patch_embed(data, octree, depth)
        # Refine local features with standard octree attention
        if self.downsample_input_embeddings:
            depth = depth - self.stem_down   # current octree depth
        octree = OctreeT(octree, self.patch_size, self.dilation, self.nempty,
                         max_depth=depth, start_depth=depth-self.num_stages+1,
                         ct_size=self.ct_size, rt_class_token=self.rt_class_token,
                         ADaPE_mode=self.ADaPE_mode,
                         ADaPE_use_accurate_point_stats=self.ADaPE_use_accurate_point_stats,
                         num_pyramid_levels=self.num_pyramid_levels,
                         num_octf_levels=self.num_octf_levels)
        octree.build_t(points=points)
        octf_feats_and_attn_maps = {}  # Store feats and attn maps per octf stage
        local_feat_dict = {}
        for i in range(self.num_octf_levels):
            data, octf_feats_and_attn_maps_i = self.octf_stage[i](data, octree, depth)
            local_feat_dict[depth] = data
            octf_feats_and_attn_maps[depth] = octf_feats_and_attn_maps_i
            data = self.downsample[i](data, octree, depth)
            depth = depth - 1

        # Compute Hierarchical Octree Attention with multi-scale Relay Tokens
        hotf_local_feat_dict, relay_token_dict, hotf_feats_and_attn_maps = (
            self.hotf_stage(data, octree, depth)
        )
        local_feat_dict.update(hotf_local_feat_dict)
        feats_and_attn_maps = {'octformer': octf_feats_and_attn_maps,
                               'hotformer': hotf_feats_and_attn_maps}
        return local_feat_dict, relay_token_dict, octree, feats_and_attn_maps


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
        num_octf_levels: int = 1,
        patch_size: int = 32,
        dilation: int = 4,
        drop_path: float = 0.5,
        nempty: bool = True,
        stem_down: int = 2,
        num_top_down: int = 2,
        fpn_channel: int = 168,
        rt_size: int = 1,
        rt_propagation: bool = False,
        rt_propagation_scale: Optional[float] = None,
        rt_init_type: str = 'avg_pool',
        rt_rpe_init: bool = False, 
        rt_class_token: bool = False,
        disable_rt: bool = False,
        ADaPE_mode: Optional[str] = None,
        ADaPE_use_accurate_point_stats: bool = False,
        grad_checkpoint: bool = True,
        downsample_input_embeddings: bool = True,
        disable_RPE: bool = False,
        conv_norm: str = 'batchnorm',
        layer_scale: Optional[float] = None,
        qkv_init: List = ['trunc_normal', 0.02],
        xcpe: bool = False,
        return_feats_and_attn_maps: bool = False,
        **kwargs
    ):
        """
        Args:
            in_channels: Number of input channels, typically 3 if only using x,y,z information.
            channels: List containing number of feature channels per stage.
            num_blocks: List containing number of OctFormer blocks per stage.
            num_heads: List containing number of attention heads per stage, defaults to channel_size//16.
            num_pyramid_levels: Number of octree levels to consider for hierarchical attention.
            num_octf_levels: Number of octformer levels to process local features before hierarchical attention
            patch_size: Size of local attention patches/windows, constructed using z-order curve traversal.
            dilation: Dilation amount for Octree attention
            drop_path: Stochastic depth probability (this is the max value stochastic depth scales to).
            nempty: Boolean indicating if only non-empty octants should be used (set True for sparse operation).
            rt_size: Size of relay tokens, note that patch_size must be divisible by this.
            rt_propagation: Boolean indicating if relay token features should be propagated to local features at the end of the stage.
            rt_propagation_scale: Learnable scalar multiplier for rt propagation step, to prevent 'blurring' of local features.
            rt_init_type: Type of initialisation to use for relay tokens. Valid types include ['avg_pool', 'max_pool', 'learnable']
            rt_class_token: Use a class token for RTSA
            disable_rt: Disable all relay token components, and process HOTFormerLoc with solely local attention (with dilation re-enabled).
            ADaPE_mode: Use Absolute Distribution-aware Position Encoding (ADaPE) during carrier token attention. Mode (valid values: ['pos','var','cov']) determines whether position, variance, or covariance is used (cumulative aggregation of those three)
            ADaPE_use_accurate_point_stats: Use accurate point statistics when computing ADaPE (by default just takes octant centroids instead of considering true point distribution)
            num_top_down: Number of top-down layers in FPN. Output features will be at Octree depth d = octree_depth - stem_down - (num_stages - 1) + num_top_down.
            fpn_channel: Number of channels in FPN top-down branch, default is to set this equal to number of channels used in Pooling (i.e. output_dim param).
            grad_checkpoint: Use gradient checkpoint to save memory, at cost of extra computation time.
            downsample_input_embeddings: Do downsampling in input conv embedding.
            disable_RPE: Disable RPE during self-attention (only applies to local attention).
            conv_norm: Type of normalisation used after convolution layers, valid params are in ['batchnorm', 'layernorm', 'powernorm'].
            layer_scale: Coefficient to initialise learnable channel-wise scale multipliers for attention outputs, or None to disable this.
            qkv_init: Method of initialisation to use for qkv linear layers
            xcpe: Use xCPE instead of CPE (from PointTransformerV3)
            return_feats_and_attn_maps: Outputs feats and attn maps from final block of each HOTFormerLoc stage
        """
        super().__init__()
        self.backbone = HOTFormerBase(
            in_channels=in_channels,
            channels=channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_pyramid_levels=num_pyramid_levels,
            num_octf_levels=num_octf_levels,
            patch_size=patch_size,
            dilation=dilation,
            drop_path=drop_path,
            nempty=nempty,
            stem_down=stem_down,
            rt_size=rt_size,
            rt_propagation=rt_propagation,
            rt_propagation_scale=rt_propagation_scale,
            rt_init_type=rt_init_type,
            rt_rpe_init=rt_rpe_init, 
            rt_class_token=rt_class_token,
            disable_rt=disable_rt,
            ADaPE_mode=ADaPE_mode,
            ADaPE_use_accurate_point_stats=ADaPE_use_accurate_point_stats,
            grad_checkpoint=grad_checkpoint,
            downsample_input_embeddings=downsample_input_embeddings,
            disable_RPE=disable_RPE,
            conv_norm=conv_norm,
            layer_scale=layer_scale,
            xcpe=xcpe,
            return_feats_and_attn_maps=return_feats_and_attn_maps
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

    def forward(self, data: Tensor, batch: Dict, depth: int):
        assert 'octree' in batch and isinstance(batch['octree'], Octree), 'Invalid batch'
        local_feat_dict, relay_token_dict, octree_t, feats_and_attn_maps = (
            self.backbone(data, batch, depth)
        )
        return local_feat_dict, relay_token_dict, octree_t, feats_and_attn_maps
