"""
HOTFormerLoc class.
Author: Ethan Griffiths
CSIRO Data61

Code adapted from OctFormer: Octree-based Transformers for 3D Point Clouds
by Peng-Shuai Wang.
"""

import torch
import torch.nn.functional as F
import ocnn

from models.octree import OctreeT
from models.hotformerloc import HOTFormerLoc
from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)


# TODO: Adapt for metric localisation (using existing HOTFormerLoc wrapper)
class HOTFormerMetricLoc(torch.nn.Module):
    def __init__(
        self,
        hotformerloc_global: HOTFormerLoc,
        coarse_feat_refiner: torch.nn.Module,
        coarse_idx: int = -1,
        fine_idx: int = -3,
        return_feats_and_attn_maps: bool = False
    ):
        """
        Class for HOTFormerLoc-based metric localisation, with coarse-to-fine
        registration inspired by GeoTransformer.

        Args:
            hotformerloc_global (nn.Module): HOTFormerLoc instance for extracting local features and global descriptor for place rec.
            coarse_feat_refiner (nn.Module): GeoTransformer (or other) instance for refining coarse features and correspondences.
            coarse_idx (int): Index corresponding to depth of coarse features, ranging from [0, num_pyramid_levels)
                              (sorted from finest to coarsest). Supports negative indices.
            fine_idx (int): Index corresponding to depth of fine features, ranging from [0, num_pyramid_levels)
                            (sorted from finest to coarsest). Supports negative indices.
            return_feats_and_attn_maps (bool): Returns intermediate features and attention maps from the backbone.

        Returns:
            model_out (dict): Dict containing outputs for local and global stages

        """
        super().__init__()
        self.hotformerloc_global = hotformerloc_global
        self.coarse_feat_refiner = coarse_feat_refiner
        self.coarse_idx = coarse_idx
        self.fine_idx = fine_idx
        self.return_feats_and_attn_maps = return_feats_and_attn_maps
        
    def forward(self, batch, global_only=False, **kwargs):
        output_dict = {}

        # 1. Extract global and local descriptors with HOTFormerLoc
        # TODO: process src and tgt in same batch or need to process separately
        global_out = self.hotformerloc_global(batch)
        output_dict.update(global_out)
        if global_only:
            return output_dict

        octree = output_dict['octree']
        local_depths = global_out['local'].keys()
        depth_coarse = local_depths[self.coarse_idx]
        depth_fine = local_depths[self.fine_idx]
        if depth_coarse >= depth_fine:
            err_str = ('Coarse feature depth in octree must be less than fine'
                       ' feature depth, check idx parameters.')
            raise ValueError(err_str)

        # TODO: Get coarse and fine feat point coords
        points_coarse = None
        points_fine = None
        feats_coarse = global_out['local'][depth_coarse]
        feats_fine = global_out['local'][depth_fine]

        # TODO: Separate reference and source feats/points, and process in GeoTrans
        ref_feats_coarse, src_feats_coarse = self.coarse_feat_refiner(
            ref_points_coarse,
            src_points_coarse,
            ref_feats_coarse,
            src_feats_coarse,
        )
        ref_feats_coarse_norm = F.normalize(ref_feats_coarse.squeeze(0), p=2, dim=1)
        src_feats_coarse_norm = F.normalize(src_feats_coarse.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_coarse'] = ref_feats_coarse_norm
        output_dict['src_feats_coarse'] = src_feats_coarse_norm

        ########################################################################
        # GeoTransformer Code
        ########################################################################
        
        # NOTE: step 2 should be achievable just from octree indices alone
        # 2. Generate ground truth node correspondences
        ref_point_to_node, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_fine, ref_points_coarse, self.num_points_in_patch
        )
        # if VIZ:
        #     save_filename = f'{SAVE_DIR}/seq{data_dict['seq_id']}-frame{data_dict['ref_frame']}' if SAVE_VIZ_PCL else None
        #     draw_point_to_node(ref_points_fine.detach().cpu().numpy(),
        #                        ref_points_coarse.detach().cpu().numpy(),
        #                        ref_point_to_node.detach().cpu().numpy(),
        #                        save_basepath=save_filename,
        #                        viz=not SAVE_VIZ_PCL)

        src_point_to_node, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_fine, src_points_coarse, self.num_points_in_patch
        )
        # if VIZ:
        #     draw_point_to_node(src_points_f.detach().cpu().numpy(),
        #                        src_points_c.detach().cpu().numpy(),
        #                        src_point_to_node.detach().cpu().numpy())

        ref_padded_points_f = torch.cat([ref_points_fine, torch.zeros_like(ref_points_fine[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_fine, torch.zeros_like(src_points_fine[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_coarse,
            src_points_coarse,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # NOTE: This is done before step 2 now.
        # # 3. Conditional Transformer
        # ref_feats_c = feats_c[:ref_length_c]
        # src_feats_c = feats_c[ref_length_c:]
        # ref_feats_c, src_feats_c = self.transformer(
        #     ref_points_coarse.unsqueeze(0),
        #     src_points_coarse.unsqueeze(0),
        #     ref_feats_c.unsqueeze(0),
        #     src_feats_c.unsqueeze(0),
        # )
        # ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        # src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        # output_dict['ref_feats_c'] = ref_feats_c_norm
        # output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_coarse_norm, src_feats_coarse_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict

    # TODO: fix for hotformermetricloc
    def print_info(self):
        print('Model class: HOTFormerMetricLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.hotformerloc_global.backbone.parameters()])
        # Backbone
        print(f'Backbone: {type(self.hotformerloc_global.backbone).__name__}\t#parameters: {n_params}')
        base_model = self.hotformerloc_global.backbone.backbone
        n_params = sum([param.nelement() for param in base_model.patch_embed.parameters()])
        print(f"  ConvEmbed:\t#parameters: {n_params}")
        n_params = sum([param.nelement() for param in base_model.octf_stage.parameters()])
        n_params += sum([param.nelement() for param in base_model.downsample.parameters()])
        print(f"  OctF Layers:\t#parameters: {n_params}")
        n_params = sum([param.nelement() for param in base_model.hotf_stage.parameters()])
        print(f"  HOTF Layers:\t#parameters: {n_params}")    
        # Pooling
        n_params = sum([param.nelement() for param in self.hotformerloc_global.pooling.parameters()])
        print(f'Pooling method: {self.hotformerloc_global.pooling.pool_method}\t#parameters: {n_params}')
        print('# channels from the backbone: {}'.format(self.hotformerloc_global.pooling.in_dim))
        print('# output channels : {}'.format(self.hotformerloc_global.pooling.output_dim))
        print(f'Embedding normalization: {self.hotformerloc_global.normalize_embeddings}')
        print('TODO: Print params for metric loc branch')
