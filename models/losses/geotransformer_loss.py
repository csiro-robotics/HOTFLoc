"""
Adaptation of loss functions used in GeoTransformer.

Ethan Griffiths (Data61, Pullenvale)
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn

from geotransformer.modules.ops import apply_transform, pairwise_distance
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.modules.loss import WeightedCircleLoss
from misc.utils import TrainingParams


class CoarseMatchingLoss(nn.Module):
    def __init__(self, params: TrainingParams):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            params.coarse_loss.positive_margin,
            params.coarse_loss.negative_margin,
            params.coarse_loss.positive_optimal,
            params.coarse_loss.negative_optimal,
            params.coarse_loss.log_scale,
        )
        self.positive_overlap = params.coarse_loss.positive_overlap
        self.eps = 1e-12

    def forward(self, output_dict):
        anc_feats = output_dict['anc_feats_coarse']
        pos_feats = output_dict['pos_feats_coarse']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(anc_feats, pos_feats, normalized=True) + self.eps)

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float() + self.eps)

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, params: TrainingParams):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = params.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        anc_node_corr_knn_points = output_dict['anc_node_corr_knn_points']
        pos_node_corr_knn_points = output_dict['pos_node_corr_knn_points']
        anc_node_corr_knn_masks = output_dict['anc_node_corr_knn_masks']
        pos_node_corr_knn_masks = output_dict['pos_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        anc_node_corr_knn_points = apply_transform(anc_node_corr_knn_points, transform)
        dists = pairwise_distance(anc_node_corr_knn_points, pos_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(anc_node_corr_knn_masks.unsqueeze(2), pos_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), anc_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), pos_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class OverallLoss(nn.Module):
    def __init__(self, params: TrainingParams):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(params)
        self.fine_loss = FineMatchingLoss(params)
        self.weight_coarse_loss = params.local.weight_coarse_loss
        self.weight_fine_loss = params.local.weight_fine_loss
        self.eval_metrics = Evaluator(params)

    def forward(self, output_dict, data_dict) -> Tuple[torch.Tensor, Dict]:
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        stats = {
            'loss': loss.item(),
            'coarse_loss': coarse_loss.item(),
            'fine_loss': fine_loss.item(),
        }
        eval_results = self.eval_metrics(output_dict, data_dict)
        stats.update(eval_results)

        return loss, stats


class Evaluator(nn.Module):
    def __init__(self, params: TrainingParams):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = params.local.acceptance_overlap
        self.acceptance_radius = params.local.acceptance_radius
        self.rre_threshold = params.local.rre_threshold
        self.rte_threshold = params.local.rte_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        anc_length_coarse = output_dict['anc_points_coarse'].shape[0]
        pos_length_coarse = output_dict['pos_points_coarse'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_anc_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_pos_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(size=(anc_length_coarse, pos_length_coarse)).cuda()
        gt_node_corr_map[gt_anc_node_corr_indices, gt_pos_node_corr_indices] = 1.0

        anc_node_corr_indices = output_dict['anc_node_corr_indices']
        pos_node_corr_indices = output_dict['pos_node_corr_indices']

        precision = gt_node_corr_map[anc_node_corr_indices, pos_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        anc_corr_points = output_dict['anc_corr_points']
        pos_corr_points = output_dict['pos_corr_points']
        anc_corr_points = apply_transform(anc_corr_points, transform)
        corr_distances = torch.linalg.norm(anc_corr_points - pos_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        rre, rte = isotropic_transform_error(transform, est_transform)
        recall = torch.logical_and(torch.lt(rre, self.rre_threshold), torch.lt(rte, self.rte_threshold)).float()
        return rre, rte, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, recall = self.evaluate_registration(output_dict, data_dict)

        # TODO: Decide whether to detach from tensors here? (tensors_to_numbers can handle for now)
        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RR': recall,
        }
