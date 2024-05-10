"""
Author: Ethan Griffiths
CSIRO Data61

Code adapted from OctFormer: Octree-based Transformers for 3D Point Clouds
by Peng-Shuai Wang.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ocnn

from models.layers.pooling_wrapper import PoolingWrapper


class OctFormerLoc(torch.nn.Module):
    def __init__(self, backbone: nn.Module, pooling: PoolingWrapper,
                 normalize_embeddings: bool = False, input_features='P'):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.input_features = input_features
        self.stats = {}
        
    def get_input_feature(self, octree):
        octree_feature = ocnn.modules.InputFeature(self.input_features, nempty=True)  # P for global position, D for local displacement (check docs)
        data = octree_feature(octree)
        return data

    def forward(self, batch):
        octree = batch['octree']
        data = self.get_input_feature(octree)
        
        x, output_depth = self.backbone(data=data, octree=octree, depth=octree.depth)
        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.pooling.in_dim, f'Backbone output tensor has: {x.shape[1]} channels. ' \
                                                  f'Expected: {self.pooling.in_dim}'
        x = self.pooling(x, octree=octree, depth=output_depth)
        if hasattr(self.pooling, 'stats'):
            self.stats.update(self.pooling.stats)

        #x = x.flatten(1)
        assert x.dim() == 2, f'Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions.'
        assert x.shape[1] == self.pooling.output_dim, f'Output tensor has: {x.shape[1]} channels. ' \
                                                      f'Expected: {self.pooling.output_dim}'

        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)

        # x is (batch_size, output_dim) tensor
        return {'global': x}

    def print_info(self):
        print('Model class: OctFormer')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f'Backbone: {type(self.backbone).__name__} #parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f'Pooling method: {self.pooling.pool_method}   #parameters: {n_params}')
        print('# channels from the backbone: {}'.format(self.pooling.in_dim))
        print('# output channels : {}'.format(self.pooling.output_dim))
        print(f'Embedding normalization: {self.normalize_embeddings}')
