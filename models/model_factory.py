# Warsaw University of Technology

import torch.nn as nn

from models.minkloc import MinkLoc
from models.octformerloc import OctFormerLoc
from models.octformer_backbone import OctFormer
from models.hotformerloc import HOTFormerLoc
from models.hotformerloc_backbone import HOTFormer
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

    if model_params.model.lower() == 'minkloc':
        block_module = create_resnet_block(model_params.block)
        backbone = MinkFPN(
            in_channels=in_channels,
            out_channels=model_params.feature_size,
            num_top_down=model_params.num_top_down,
            conv0_kernel_size=model_params.conv0_kernel_size,
            block=block_module,
            layers=model_params.layers,
            planes=model_params.planes
        )
        pooling = PoolingWrapper(
            pool_method=model_params.pooling,
            in_dim=model_params.feature_size,
            output_dim=model_params.output_dim
        )
        model = MinkLoc(
            backbone=backbone,
            pooling=pooling,
            normalize_embeddings=model_params.normalize_embeddings
        )
    elif 'hotformerloc' in model_params.model.lower():
        in_channels = get_in_channels(model_params.input_features)
        backbone = HOTFormer(
            in_channels=in_channels,
            channels=model_params.channels,
            num_blocks=model_params.num_blocks,
            num_heads=model_params.num_heads,
            num_pyramid_levels=model_params.num_pyramid_levels,
            num_octf_levels=model_params.num_octf_levels,
            patch_size=model_params.patch_size,
            dilation=model_params.dilation,
            drop_path=model_params.drop_path,
            stem_down=model_params.num_input_downsamples,
            num_top_down=model_params.num_top_down,
            fpn_channel=model_params.feature_size,
            rt_size=model_params.ct_size,
            rt_propagation=model_params.ct_propagation,
            rt_propagation_scale=model_params.ct_propagation_scale,
            ADaPE_mode=model_params.ADaPE_mode,
            grad_checkpoint=model_params.grad_checkpoint,
            downsample_input_embeddings=model_params.downsample_input_embeddings,
            disable_RPE=model_params.disable_RPE,
            conv_norm=model_params.conv_norm,
            layer_scale=model_params.layer_scale,
            qkv_init=model_params.qkv_init,
            xcpe=model_params.xcpe,
        )
        pooling = PoolingWrapper(
            pool_method=model_params.pooling,
            in_dim=model_params.feature_size,
            output_dim=model_params.output_dim,
            num_pyramid_levels=model_params.num_pyramid_levels,
            k_pooled_tokens=model_params.k_pooled_tokens,
        )
        model = HOTFormerLoc(
            backbone=backbone,
            pooling=pooling,
            normalize_embeddings=model_params.normalize_embeddings,
            input_features=model_params.input_features
        )
    elif any(model in model_params.model.lower() for model in ('octformer', 'hotformer')):
        in_channels = get_in_channels(model_params.input_features)
        backbone = OctFormer(
            in_channels=in_channels,
            channels=model_params.channels,
            num_blocks=model_params.num_blocks,
            num_heads=model_params.num_heads,
            ct_layers=model_params.ct_layers,
            patch_size=model_params.patch_size,
            dilation=model_params.dilation,
            drop_path=model_params.drop_path,
            fpn_channel=model_params.feature_size,
            num_top_down=model_params.num_top_down,
            stem_down=model_params.num_input_downsamples,
            ct_size=model_params.ct_size,
            ct_propagation=model_params.ct_propagation,
            ct_propagation_scale=model_params.ct_propagation_scale,
            ADaPE_mode=model_params.ADaPE_mode,
            downsample_input_embeddings=model_params.downsample_input_embeddings,
            grad_checkpoint=model_params.grad_checkpoint,
            disable_RPE=model_params.disable_RPE,
            conv_norm=model_params.conv_norm,
            layer_scale=model_params.layer_scale,
            qkv_init=model_params.qkv_init,
            xcpe=model_params.xcpe,
        )
        pooling = PoolingWrapper(
            pool_method=model_params.pooling,
            in_dim=model_params.feature_size,
            output_dim=model_params.output_dim
        )
        model = OctFormerLoc(
            backbone=backbone,
            pooling=pooling,
            normalize_embeddings=model_params.normalize_embeddings,
            input_features=model_params.input_features
        )
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
