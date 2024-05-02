# Warsaw University of Technology

import torch.nn as nn

from models.minkloc import MinkLoc
from models.octformer_pr import OctFormerPR
from models.octformer_backbone import OctFormer
from misc.utils import ModelParams
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from models.layers.eca_block import ECABasicBlock
from models.minkfpn import MinkFPN
from models.layers.pooling_wrapper import PoolingWrapper

def get_in_channels(input_features: str) -> int:
    in_channels = 0
    channel_num_dict = {'L': 3, 'P': 3, 'D': 1, 'N': 3}  # https://ocnn-pytorch.readthedocs.io/en/latest/modules/octree.html#ocnn.octree.Octree.get_input_feature
    
    for feature in input_features:
        assert feature in channel_num_dict.keys(), "Invalid input features specified, must be in ['L','P','D','N']"
        in_channels += channel_num_dict[feature]

    assert in_channels > 0, "Invalid input features specified, must be in ['L','P','D','N']"
    
    return in_channels

def model_factory(model_params: ModelParams):
    in_channels = 1

    if model_params.model == 'MinkLoc':
        block_module = create_resnet_block(model_params.block)
        backbone = MinkFPN(in_channels=in_channels, out_channels=model_params.feature_size,
                           num_top_down=model_params.num_top_down, conv0_kernel_size=model_params.conv0_kernel_size,
                           block=block_module, layers=model_params.layers, planes=model_params.planes)
        pooling = PoolingWrapper(pool_method=model_params.pooling, in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)
        model = MinkLoc(backbone=backbone, pooling=pooling, normalize_embeddings=model_params.normalize_embeddings)
    elif model_params.model == 'OctFormer':
        in_channels = get_in_channels(model_params.input_features)
        backbone = OctFormer(in_channels=in_channels, channels=model_params.channels, num_blocks=model_params.num_blocks,
                             num_heads=model_params.num_heads, patch_size=model_params.patch_size, fpn_channel=model_params.feature_size,
                             num_top_down=model_params.num_top_down)
        pooling = PoolingWrapper(pool_method=model_params.pooling, in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)
        model = OctFormerPR(backbone=backbone, pooling=pooling, normalize_embeddings=model_params.normalize_embeddings,
                            input_features=model_params.input_features)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(model_params.model))

    return model


def create_resnet_block(block_name: str) -> nn.Module:
    if block_name == 'BasicBlock':
        block_module = BasicBlock
    elif block_name == 'Bottleneck':
        block_module = Bottleneck
    elif block_name == 'ECABasicBlock':
        block_module = ECABasicBlock
    else:
        raise NotImplementedError('Unsupported network block: {}'.format(block_name))

    return block_module
