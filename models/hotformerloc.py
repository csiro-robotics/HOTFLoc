"""
HOTFormerLoc class.
Author: Ethan Griffiths
CSIRO Data61

Code adapted from OctFormer: Octree-based Transformers for 3D Point Clouds
by Peng-Shuai Wang.
"""
from typing import List, Set, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import ocnn

from models.layers.pooling_wrapper import PoolingWrapper
from models.octree import OctreeT


class HOTFormerLoc(torch.nn.Module):
    def __init__(self, backbone: nn.Module, pooling: PoolingWrapper,
                 normalize_embeddings: bool = False, input_features='P',
                 return_feats_and_attn_maps: bool = False,
                 reranking_mode: Optional[str] = None):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.input_features = input_features
        self.return_feats_and_attn_maps = return_feats_and_attn_maps
        if reranking_mode not in (None, 'relay_token_gc'):
            raise ValueError('Invalid re-ranking mode')
        self.reranking_mode = reranking_mode
        self.stats = {}
        
    def get_input_feature(self, octree):
        if self.input_features.upper() == 'F':
            # Make input have unit features, similar to in MinkLoc
            data = torch.ones(size=(octree.nnum_nempty[octree.depth],1),
                              device=octree.device, dtype=torch.float32)
            assert len(data) == len(octree.points[octree.depth])
        else:
            octree_feature = ocnn.modules.InputFeature(self.input_features, nempty=True)  # P for global position, L for local position (check docs)
            data = octree_feature(octree)
        return data

    def forward(self, batch, **kwargs):
        octree = batch['octree']
        data = self.get_input_feature(octree)

        local_feat_dict, relay_token_dict, octree, feats_and_attn_maps = (
            self.backbone(data=data, batch=batch, depth=octree.depth)
        )
        if self.pooling.pooled_feats == 'local':
            x = local_feat_dict
        elif self.pooling.pooled_feats == 'relaytokens':
            x = relay_token_dict
        else:
            raise ValueError(f'Invalid option for pooled features: '
                             f'\'{self.pooling.pooled_feats}\'')
        x = self.pooling(x, octree=octree)
        if hasattr(self.pooling, 'stats'):
            self.stats.update(self.pooling.stats)

        # x is (batch_size, output_dim) tensor
        assert x.dim() == 2, f'Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions.'
        assert x.shape[1] == self.pooling.output_dim, f'Output tensor has: {x.shape[1]} channels. ' \
                                                      f'Expected: {self.pooling.output_dim}'

        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)

        octf_qkv_std, hosa_qkv_std, rt_qkv_std = self.get_qkv_std(feats_and_attn_maps, octree)

        return_dict = {'global': x, 'local': local_feat_dict, 'rt': relay_token_dict,
                       'octree': octree, 'octf_qkv_std': octf_qkv_std,
                       'hosa_qkv_std': hosa_qkv_std, 'rt_qkv_std': rt_qkv_std,
                       'rt_final_cls_attn_vals': feats_and_attn_maps['hotformer'][-1]['rt_final_cls_attn_vals']}
        if self.return_feats_and_attn_maps:
            return_dict.update({'feats_and_attn_maps': feats_and_attn_maps['hotformer'],
                                'octf_feats_and_attn_maps': feats_and_attn_maps['octformer']})
        return return_dict

    def rerank(self, query_output: dict, nn_outputs: List[dict]):
        """
        Perform re-ranking of query and N candidates. Note that only 'octree',
        'rt', and 'rt_final_cls_attn_vals' keys are needed from HOTFormerLoc outputs.

        Args:
            query_output (dict): HOTFormerLoc output for query submap
            nn_outputs (List[dict]): List of HOTFormerLoc outputs for each nn submap
        """
        # TODO: Separate this into a separate class, and init that class in __init__
        if self.reranking_mode == 'relay_token_gc':
            # Relay token geometric consistency
            from models.relay_token_utils import concat_and_pad_rt, unpad_and_split_rt
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
            concat_and_pad_rt(relay_token_dict, octree)
            pass
        else:
            raise NotImplementedError
    
    @staticmethod
    def get_qkv_std(feats_and_attn_maps: dict, octree: OctreeT) -> Tuple[dict, dict, list]:
        """
        Returns standard deviation of query, key, value outputs for each block of
        each HOTFormer pyramid level.
        """
        # Separate `qkv_std` from `feats_and_attn_maps`
        octf_qkv_std = {}
        hosa_qkv_std = {depth: [] for depth in octree.pyramid_depths}
        rt_qkv_std = []
        for depth, octf_feats_and_attn_maps_i in feats_and_attn_maps['octformer'].items():
            octf_qkv_std[depth] = []
            for dict_i in octf_feats_and_attn_maps_i:
                if 'local_qkv_std' in dict_i.keys():
                    octf_qkv_std[depth].append(dict_i.pop('local_qkv_std'))
        # Invert dict structure so outside is dict indexed by depth, and each depth contains a list of qkv std per block
        for dict_i in feats_and_attn_maps['hotformer']:
            if 'local_qkv_std' in dict_i.keys():
                hosa_qkv_std_temp_i = dict_i.pop('local_qkv_std')
                for depth in hosa_qkv_std_temp_i.keys():
                    hosa_qkv_std[depth].append(hosa_qkv_std_temp_i[depth])
            if 'rt_qkv_std' in dict_i.keys():
                rt_qkv_std.append(dict_i.pop('rt_qkv_std'))
        return octf_qkv_std, hosa_qkv_std, rt_qkv_std

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Set of parameters that should not use weight decay."""
        return {'rpe_table', 'rt_init_token', 'rt_cls_token'}

    def print_info(self):
        print('Model class: HOTFormerLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        # Backbone
        print(f'Backbone: {type(self.backbone).__name__}\t#parameters: {n_params}')
        base_model = self.backbone.backbone
        n_params = sum([param.nelement() for param in base_model.patch_embed.parameters()])
        print(f"  ConvEmbed:\t#parameters: {n_params}")
        n_params = sum([param.nelement() for param in base_model.octf_stage.parameters()])
        n_params += sum([param.nelement() for param in base_model.downsample.parameters()])
        print(f"  OctF Layers:\t#parameters: {n_params}")
        n_params = sum([param.nelement() for param in base_model.hotf_stage.parameters()])
        print(f"  HOTF Layers:\t#parameters: {n_params}")    
        # Pooling
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f'Pooling method: {self.pooling.pool_method}\t#parameters: {n_params}')
        print('# channels from the backbone: {}'.format(self.pooling.in_dim))
        print('# output channels : {}'.format(self.pooling.output_dim))
        print(f'Embedding normalization: {self.normalize_embeddings}')
