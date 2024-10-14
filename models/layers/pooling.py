# Pooling methods code based on: https://github.com/filipradenovic/cnnimageretrieval-pytorch

from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
import ocnn
import MinkowskiEngine as ME
from ocnn.octree import Octree
from models.octree import OctreeT

from models.layers.netvlad import NetVLADLoupe, GatingContext
from models.layers.pyramid_netvlad import PyramidNetVLAD
from models.layers.salsa import AdaptivePooling, Mixer
from models.layers.octformer_layers import MLP
from models.relay_token_utils import concat_and_pad_rt

class MAC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class SPoC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class GeM(nn.Module):
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        #temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p),
                               coordinate_manager = x.coordinate_manager,
                               coordinate_map_key = x.coordinate_map_key)
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1./self.p)    # Return (batch_size, n_features) tensor


class OctGeM(nn.Module):
    """
    Octree compatible version of GeM pooling.
    """
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(OctGeM, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ocnn.nn.OctreeGlobalPool(nempty=True)

    def forward(self, x: Tensor, octree: Octree, depth: int):
        # This implicitly applies ReLU on x (clamps negative values)
        temp = x.clamp(min=self.eps).pow(self.p)
        temp = self.f(temp, octree, depth)  # Apply GlobalAvgPooling
        return temp.pow(1./self.p)          # Return (batch_size, n_features) tensor

class RelayTokenGeM(OctGeM):
    """
    GeM pooling compatible with multi-scale relay tokens (or really any
    batched tensor input.)
    """
    def __init__(self, input_dim, p=3, eps=1e-6):
        super().__init__(input_dim=input_dim, p=p, eps=eps)
        self.f = None
    
    def forward(self, x: Tensor):  # x: (B, N, C)
        # This implicitly applies ReLU on x (clamps negative values)
        temp = x.clamp(min=self.eps).pow(self.p)
        temp = torch.mean(temp, dim=1)  # Apply GlobalAvgPooling
        return temp.pow(1./self.p)      # Return (batch_size, n_features) tensor


class NetVLADWrapper(nn.Module):
    def __init__(self, feature_size, output_dim, gating=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.net_vlad = NetVLADLoupe(feature_size=feature_size, cluster_size=64, output_dim=output_dim, gating=gating,
                                     add_batch_norm=True)

    def forward(self, x: ME.SparseTensor):
        # x is (batch_size, C, H, W)
        assert x.F.shape[1] == self.feature_size
        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        batch_size = len(features)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros

        x = self.net_vlad(features)
        assert x.shape[0] == batch_size
        assert x.shape[1] == self.output_dim
        return x    # Return (batch_size, output_dim) tensor

class PyramidOctGeMWrapper(nn.Module):
    def __init__(self, input_dim, output_dim, num_pyramid_levels: int, p=3,
                 eps=1e-6, gating=False, add_batch_norm=True):
        super().__init__()
        self.input_dim = input_dim
        assert num_pyramid_levels > 0, "Minimum 1 pyramid layer"
        self.num_pyramid_levels = num_pyramid_levels
        # Same output number of channels as input number of channels
        self.output_dim = output_dim
        self.p = nn.Parameter(torch.ones(num_pyramid_levels) * p)
        self.eps = eps
        self.gating = gating
        self.f = ocnn.nn.OctreeGlobalPool(nempty=True)
        self.linear_bn = nn.Sequential(
            nn.Linear(
                input_dim*num_pyramid_levels, output_dim, bias=False
            ),
            nn.BatchNorm1d(input_dim),
        )
        if self.gating:
            self.context_gating = GatingContext(output_dim,
                                                add_batch_norm=add_batch_norm)

    def forward(self, local_feat_dict: Dict[int, Tensor], octree: OctreeT,
                depth: int = None):
        # Generate global descriptor for each pyramid level
        pyramid_descriptors = []
        for j, depth_j in enumerate(local_feat_dict.keys()):
            temp = local_feat_dict[depth_j].clamp(min=self.eps).pow(self.p[j])
            temp = self.f(temp, octree, depth_j)  # Apply GlobalAvgPooling
            pyramid_descriptors.append(temp.pow(1./self.p[j]))  # (batch_size, n_features) tensor
        
        # Concat and fuse into a single descriptor
        global_descriptor = torch.cat(pyramid_descriptors, dim=-1)
        global_descriptor = self.linear_bn(global_descriptor)

        if self.gating:
            global_descriptor = self.context_gating(global_descriptor)
        
        return global_descriptor

# TODO
class PyramidNetVLADWrapper(nn.Module):
    """
    Wrapper for PyramidNetVLAD, based on PPT-Net.
    """
    def __init__(self, feature_size, output_dim, cluster_size=[64, 16, 4],
                 gating=True, add_batch_norm=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.cluster_size = cluster_size
        self.pyramid_netvlad = PyramidNetVLAD(
            feature_size, output_dim, cluster_size, gating, add_batch_norm
        )

    def forward(self, local_feat_dict: Dict[int, Tensor], depth: int = None):
        pass
        

class AttnPoolWrapper(nn.Module):
    """
    Wrapper for adaptive attention pooling + MLP token mixer, inspired by
    SALSA: https://arxiv.org/pdf/2407.08260. Also allows using GeM instead of
    token mixer.
    """
    def __init__(self, feature_size: int = 256, output_dim: int = 256,
                 k_pooled_tokens: int = 64, mlp_ratio: int = 1,
                 aggregator: str = 'mixer', mix_depth: int = 4):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.k_pooled_tokens = k_pooled_tokens
        self.mlp_ratio = mlp_ratio
        self.aggregator = aggregator
        self.attpool = AdaptivePooling(
            feature_dim=feature_size,         # originally 512
            k_pooled_tokens=k_pooled_tokens,  # originally 16
        )
        if aggregator.lower() == 'mixer':
            # TODO: Currently these values are based on a fixed ratio to ensure
            #       the output is equal to output_dim, but may be worth trying
            #       different ratios of tokens to channels in the MLP mixer.
            k_output_tokens = k_pooled_tokens // 4  # originally 128
            out_d = output_dim // k_output_tokens   # originally 4
            self.descriptor_extractor = Mixer(
                k_input_tokens=k_pooled_tokens,
                k_output_tokens=k_output_tokens,
                in_d=feature_size,
                mix_depth=mix_depth,
                mlp_ratio=mlp_ratio,
                out_d=out_d,
            )  # output size = k_output_tokens * out_d
        elif aggregator.lower() == 'gem':
            self.token_processor = nn.Sequential(
                nn.LayerNorm(feature_size),
                MLP(
                    in_features=feature_size,
                    hidden_features=feature_size*mlp_ratio,
                    out_features=output_dim,
                ),
            )
            self.descriptor_extractor = RelayTokenGeM(input_dim=feature_size)
        else:
            raise NotImplementedError(f'No valid aggregator: {aggregator}')
            

    def forward(self, relay_token_dict: Dict[int, Tensor],
                octree: OctreeT, depth: int = None):
        split_tokens = concat_and_pad_rt(relay_token_dict, octree)
        attn_mask = self.calc_rt_attn_mask(octree.rt_attn_mask)
        # Pool to k tokens
        token_attn = self.attpool(split_tokens, attn_mask)
        # Aggregate tokens into a global descriptor
        if self.aggregator.lower() != 'mixer':
            token_attn = token_attn + self.token_processor(token_attn)
        global_descriptor = self.descriptor_extractor(token_attn)
        return global_descriptor

    def calc_rt_attn_mask(self, rt_attn_mask: Tensor) -> Tensor:
        """
        Alters relay token attention mask to be suitable for
        attentional pooling with learnable query matrix.
        """
        # All query tokens should ignore padding tokens
        attn_mask = rt_attn_mask[:, 0, :].unsqueeze(1)  # (B, N, N) -> (B, k, N)
        attn_mask = attn_mask.repeat(1, self.k_pooled_tokens, 1)
        return attn_mask