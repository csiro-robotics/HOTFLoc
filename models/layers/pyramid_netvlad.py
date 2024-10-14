"""
Pyramid Net VLAD implementation, based on PPT-Net:
https://openaccess.thecvf.com/content/ICCV2021/papers/Hui_Pyramid_Point_Cloud_Transformer_for_Large-Scale_Place_Recognition_ICCV_2021_paper.pdf.

Adapted by Ethan Griffiths (Data61, Pullenvale)
"""

import math
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.layers.netvlad import NetVLADLoupe, GatingContext

"""
NOTE: The toolbox can only pool lists of features of the same length. It was specifically optimized to efficiently
do so. One way to handle multiple lists of features of variable length is to create, via a data augmentation
technique, a tensor of shape: 'batch_size'x'max_samples'x'feature_size'. Where 'max_samples' would be the maximum
number of feature per list. Then for each list, you would fill the tensor with 0 values.
"""

class PyramidNetVLAD(nn.Module):    
    def __init__(self, feature_size: int, output_dim: int,
                 cluster_size=[64, 16, 4], gating=True, add_batch_norm=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.cluster_size = cluster_size
        # NOTE: Cluster size list length must match num pyramid levels
        self.vlad_layers = nn.ModuleList()
        for cluster_size_i in cluster_size:
            self.vlad_layers.append(
                NetVLADLoupe(
                    feature_size=feature_size, cluster_size=cluster_size_i,
                    output_dim=output_dim, gating=gating,
                    add_batch_norm=add_batch_norm,
                )
            )
        sum_cluster_size = sum(cluster_size)
        self.hidden_weights = nn.Parameter(
            torch.randn(feature_size*sum_cluster_size, output_dim)
            * (1 / math.sqrt(feature_size))
        )
        self.bn = nn.BatchNorm1d(output_dim)
        self.gating = gating
        if self.gating:
            self.context_gating = GatingContext(
                output_dim=output_dim, add_batch_norm=add_batch_norm,
            )

    def forward(self, local_feat_dict: Dict[int, Tensor]):
        assert len(local_feat_dict) == len(self.cluster_size)
        # TODO: Allow VLAD to accept Octree input (expects batched tensor)
        # v0 = self.vlad0(f0)
        # v1 = self.vlad1(f1)
        # v2 = self.vlad2(f2)
        # v3 = self.vlad3(f3)
        # vlad = torch.cat((v0, v1, v2, v3), dim=-1)
        # vlad = torch.matmul(vlad, self.hidden_weights)      # B x (1024*64) X (1024*64) x 256 -> B x 256
        # vlad = self.bn2(vlad)                               # B x 256 -> B x 256
        
        # if self.gating:
        #     vlad = self.context_gating(vlad)                # B x 256 -> B x 256
        # return vlad    
        pass