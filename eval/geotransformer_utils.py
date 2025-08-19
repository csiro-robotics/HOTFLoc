"""
Evaluate Geotransformer metrics. Adapted from:
https://github.com/qinzheng93/GeoTransformer/blob/main/experiments/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/eval.py

Ethan Griffiths (Data61, Pullenvale)
"""
import time
from typing import Dict

import numpy as np

from misc.utils import TrainingParams
from misc.point_clouds import registration_with_ransac_from_correspondences
# from geotransformer.utils.open3d import registration_with_ransac_from_correspondences
from geotransformer.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
)


def compute_geotransformer_metrics(
    output_dict: Dict, data_dict: Dict, params: TrainingParams, use_ransac=False
):
    registration_methods = ['lgr']
    if use_ransac:
        registration_methods.append('ransac')
    metrics_dict = {}
    
    anc_nodes = output_dict['anc_points_coarse']
    pos_nodes = output_dict['pos_points_coarse']
    anc_node_corr_indices = output_dict['anc_node_corr_indices']
    pos_node_corr_indices = output_dict['pos_node_corr_indices']

    anc_corr_points = output_dict['anc_corr_points']
    pos_corr_points = output_dict['pos_corr_points']

    gt_node_corr_indices = output_dict['gt_node_corr_indices']
    gt_transform = data_dict['transform']

    # Get LGR correspondence metrics
    anc_node_corr_knn_masks = output_dict['anc_node_corr_knn_masks']
    pos_node_corr_knn_masks = output_dict['pos_node_corr_knn_masks']
    num_points_per_patch = np.concatenate(
        (np.float32(anc_node_corr_knn_masks).sum(1),
         np.float32(pos_node_corr_knn_masks).sum(1),),
        axis=0,
    )
    num_corr_points = np.asarray(output_dict['num_corr_points'])
    corr_scores = output_dict['corr_scores']
    metrics_dict['num_pts_per_patch'] = num_points_per_patch.mean()
    metrics_dict['num_corr_patches_lgr'] = len(num_corr_points)
    metrics_dict['num_corr_pts_per_patch_lgr'] = num_corr_points.mean()
    metrics_dict['corr_score_lgr'] = corr_scores.mean()

    # 1. evaluate correspondences
    # 1.1 evaluate coarse correspondences
    coarse_matching_result_dict = evaluate_sparse_correspondences(
        anc_nodes,
        pos_nodes,
        anc_node_corr_indices,
        pos_node_corr_indices,
        gt_node_corr_indices,
    )

    coarse_precision = coarse_matching_result_dict['precision']

    metrics_dict['PIR'] = coarse_precision
    metrics_dict['PMR>0'] = float(coarse_precision > 0)
    metrics_dict['PMR>=0.1'] = float(coarse_precision >= 0.1)
    metrics_dict['PMR>=0.3'] = float(coarse_precision >= 0.3)
    metrics_dict['PMR>=0.5'] = float(coarse_precision >= 0.5)

    # 1.2 evaluate fine correspondences
    fine_matching_result_dict = evaluate_correspondences(
        anc_corr_points,
        pos_corr_points,
        np.linalg.inv(gt_transform),  # NOTE: Geotrans funcs expect TF from pos->anc
        positive_radius=params.local.acceptance_radius,
    )

    inlier_ratio = fine_matching_result_dict['inlier_ratio']
    overlap = fine_matching_result_dict['overlap']
    residual = fine_matching_result_dict['residual']
    num_corr = fine_matching_result_dict['num_corr']

    metrics_dict['FMR'] = float(inlier_ratio >= params.local.inlier_ratio_threshold)
    metrics_dict['IR'] = inlier_ratio
    metrics_dict['OV'] = overlap
    metrics_dict['residual'] = residual
    metrics_dict['num_corr'] = num_corr

    # 2. evaluate registration
    for method in registration_methods:
        if method == 'lgr':
            est_transform = output_dict['estimated_transform']
        elif method == 'ransac':
            # NOTE: CURRENTLY THIS ALLOWS MULTIPLE CORRESPONDENCES PER POINT.
            #       SHOULD PROBABLY FILTER CORRESPONDENCES FOR RANSAC WITH HIGHER CONFIDENCE THRESHOLD.
            #       ALSO COULD TRY EGONN STYLE - MATCH COARSE FEATS WITH RANSAC
            tic = time.time()
            est_transform = registration_with_ransac_from_correspondences(
                anc_corr_points,
                pos_corr_points,
                distance_threshold=params.local.ransac_distance_threshold,
                ransac_n=params.local.ransac_num_points,
                num_iterations=params.local.ransac_num_iterations,
            )
            metrics_dict['t_ransac'] = time.time() - tic
        else:
            raise ValueError(f'Unsupported registration method: {method}.')

        rre, rte = compute_registration_error(gt_transform, est_transform)
        accepted = rre < params.local.rre_threshold and rte < params.local.rte_threshold
        metrics_dict[f'rre_{method}'] = rre
        metrics_dict[f'rte_{method}'] = rte
        metrics_dict[f'success_{method}'] = float(accepted)

    return metrics_dict