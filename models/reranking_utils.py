
"""
Adapted from SpectralGV repo: https://github.com/csiro-robotics/SpectralGV/blob/main
"""
# Functions in this file are adapted from: https://github.com/ZhiChen902/SC2-PCR/blob/main/SC2_PCR.py
from typing import Optional

import numpy as np
import torch
from torch import Tensor

def match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features):
    # normalize:
    src_features = torch.nn.functional.normalize(src_features, p=2.0, dim=1)
    tgt_features = torch.nn.functional.normalize(tgt_features, p=2.0, dim=1)

    distance = torch.cdist(src_features, tgt_features)
    min_vals, min_ids = torch.min(distance, dim=2)
 
    min_ids = min_ids.unsqueeze(-1).expand(-1, -1, 3)
    tgt_keypts_corr = torch.gather(tgt_keypts, 1, min_ids)
    src_keypts_corr = src_keypts

    return src_keypts_corr, tgt_keypts_corr

def power_iteration(M, num_iterations=5):
    """
    Calculate the leading eigenvector using power iteration algorithm
    Input:
        - M:      [bs, num_pts, num_pts] the adjacency matrix
    Output:
        - leading_eig: [bs, num_pts] leading eigenvector
    """
    leading_eig = torch.ones_like(M[:, :, 0:1])
    leading_eig_last = leading_eig
    for i in range(num_iterations):
        leading_eig = torch.bmm(M, leading_eig)
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
        if torch.allclose(leading_eig, leading_eig_last):
            break
        leading_eig_last = leading_eig
    leading_eig = leading_eig.squeeze(-1)
    return leading_eig


def cal_spatial_consistency( M, leading_eig):
    """
    Calculate the spatial consistency based on spectral analysis.
    Input:
        - M:          [bs, num_pts, num_pts] the adjacency matrix
        - leading_eig [bs, num_pts]           the leading eigenvector of matrix M
    Output:
        - sc_score_list [bs, 1]
    """
    spatial_consistency = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
    spatial_consistency = spatial_consistency.squeeze(-1) / M.shape[1]
    return spatial_consistency


def sgv_parallel(
    src_keypts,
    tgt_keypts,
    src_features,
    tgt_features,
    d_thresh=0.4,
    return_spatial_consistency=False,
):
    """
    Input:
        - src_keypts: [1, num_pts, 3]
        - tgt_keypts: [bs, num_pts, 3]
        - src_features: [1, num_pts, D]
        - tgt_features: [bs, num_pts, D]
    Output:
        - lead_eigvec:    [bs, num_pts], leading eigenvector of spatial consistency adj mat
        - sc_score_list:   [bs, 1], spatial consistency score for each candidate
    """
    # Correspondence Estimation: Nearest Neighbour Matching
    src_keypts_corr, tgt_keypts_corr = match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features)

    # Spatial Consistency Adjacency Matrix
    src_dist = torch.norm((src_keypts_corr[:, :, None, :] - src_keypts_corr[:, None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts_corr[:, :, None, :] - tgt_keypts_corr[:, None, :, :]), dim=-1)
    cross_dist = torch.abs(src_dist - target_dist)
    adj_mat = torch.clamp(1.0 - cross_dist ** 2 / d_thresh ** 2, min=0)

    # Spatial Consistency Score
    lead_eigvec = power_iteration(adj_mat)
    if not return_spatial_consistency:
        return lead_eigvec

    sc_score_list = cal_spatial_consistency(adj_mat, lead_eigvec)
    sc_score_list = np.squeeze(sc_score_list.cpu().detach().numpy())
    return lead_eigvec, sc_score_list


def sgv_fn(query_keypoints, tgt_lfs, tgt_kps, d_thresh=5.0):
    src_keypts = query_keypoints['keypoints'].unsqueeze(0).cuda()
    src_features = query_keypoints['features'].unsqueeze(0).cuda()

    conf_list = sgv_parallel(src_keypts, tgt_kps.cuda(), src_features, tgt_lfs.cuda(), d_thresh=d_thresh)

    return  conf_list


################################################################################
# Batched versions of SGV functions 
################################################################################
def batched_match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features, mask: Optional[Tensor] = None):
# def batched_match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features, mask_zeros=False):
    # normalize:
    src_features = torch.nn.functional.normalize(src_features, p=2.0, dim=-2)
    tgt_features = torch.nn.functional.normalize(tgt_features, p=2.0, dim=-2)

    # if mask_zeros:  # Mask out features that are entirely zeros
    #     src_mask = (src_features == 0).all(dim=-1, keepdim=True)
    #     tgt_mask = (tgt_features == 0).all(dim=-1, keepdim=True)
    #     src_features.masked_fill_(src_mask, float('-inf'))
    #     tgt_features.masked_fill_(tgt_mask, float('-inf'))

    distance = torch.cdist(src_features, tgt_features)
    # if mask_zeros:  # Distance between (-inf,-inf) is nan, so need to filter
        # distance.masked_fill_(torch.isnan(distance), float('inf'))
    if mask is not None:
        distance.masked_fill_(mask.logical_not(), float('inf'))
    min_vals, min_ids = torch.min(distance, dim=-1)
 
    min_ids = min_ids.unsqueeze(-1).expand(-1, -1, -1, 3)
    tgt_keypts_corr = torch.gather(tgt_keypts, -2, min_ids)
    src_keypts_corr = src_keypts

    return src_keypts_corr, tgt_keypts_corr

def batched_power_iteration(M, num_iterations=5):
    """
    Calculate the leading eigenvector using power iteration algorithm
    Input:
        - M:      [bs, num_pts, num_pts] the adjacency matrix
    Output:
        - leading_eig: [bs, num_pts] leading eigenvector
    """
    leading_eig = torch.ones_like(M[..., 0:1])
    leading_eig_last = leading_eig
    for i in range(num_iterations):
        leading_eig = torch.matmul(M, leading_eig)
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=-2, keepdim=True) + 1e-6)
        if torch.allclose(leading_eig, leading_eig_last):
            break
        leading_eig_last = leading_eig
    leading_eig = leading_eig.squeeze(-1)
    return leading_eig


def batched_cal_spatial_consistency( M, leading_eig):
    """
    Calculate the spatial consistency based on spectral analysis.
    Input:
        - M:          [B, nn, num_pts, num_pts] the adjacency matrix
        - leading_eig [B, nn, num_pts]           the leading eigenvector of matrix M
    Output:
        - sc_score_list [B, nn, 1]
    """
    spatial_consistency = leading_eig[..., None, :] @ M @ leading_eig[..., None]
    spatial_consistency = spatial_consistency.squeeze(-1) / M.shape[-2]
    return spatial_consistency


def batched_sgv_parallel(
    src_keypts: Tensor,
    tgt_keypts: Tensor,
    src_features: Optional[Tensor] = None,
    tgt_features: Optional[Tensor] = None,
    d_thresh=0.4,
    mask: Optional[Tensor] = None,
    # mask_zeros=False,
    return_spatial_consistency=False,
):
    """
    Batched version of SGV.
    Input:
        - src_keypts: [B, 1, num_pts, 3] or [B, nn, num_pts, 3] or [B, num_pts, 3]
        - tgt_keypts: [B, nn, num_pts, 3]
        - src_features: [B, 1, num_pts, D] or [B, num_pts, D] (if None, assume keypts already sorted by correspondences)
        - tgt_features: [B, nn, num_pts, D] (if None, assume keypts already sorted by correspondences)
        - d_thresh: Distance threshold for adjacency matrix
        # - mask_zeros: Ignore feats filled with all zeros (assumes they are masked) 
        - mask: Mask of keypts to ignore of shape [B, nn, num_pts, num_pts] (True to keep, False to ignore)
    Output:
        - lead_eigvec:    [B, nn, num_pts], leading eigenvector of spatial consistency adj mat
        - sc_score_list:   [B, nn, 1], spatial consistency score for each candidate
    """
    if src_keypts.ndim == 3:
        src_keypts = src_keypts.unsqueeze(1)
    assert src_keypts.ndim == 4
    assert tgt_keypts.ndim == 4
    assert src_keypts.size(3) == 3
    assert tgt_keypts.size(3) == 3
    if src_features is not None:
        assert tgt_features is not None
        assert src_keypts.size(1) == 1
        if src_features.ndim == 3:
            src_features = src_features.unsqueeze(1)
        assert src_features.ndim == 4
        assert tgt_features.ndim == 4
        assert src_features.size(1) == 1
    else:
        assert tgt_features is None
        assert src_keypts.size(1) == tgt_keypts.size(1)

    if src_features is None:
        # Assume keypts already sorted by correspondences
        src_keypts_corr, tgt_keypts_corr = src_keypts, tgt_keypts
    else:
        # Correspondence Estimation: Nearest Neighbour Matching
        src_keypts_corr, tgt_keypts_corr = batched_match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features, mask)

    # Spatial Consistency Adjacency Matrix
    src_dist = torch.norm((src_keypts_corr[..., None, :] - src_keypts_corr[..., None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts_corr[..., None, :] - tgt_keypts_corr[..., None, :, :]), dim=-1)
    cross_dist = torch.abs(src_dist - target_dist)
    adj_mat = torch.clamp(1.0 - cross_dist ** 2 / d_thresh ** 2, min=0)

    # Mask out padded keypts
    if mask is not None:
        # Keep diagonal to ensure consistency
        diag = torch.ones_like(torch.diagonal(mask, dim1=-2, dim2=-1))
        mask_ignore_diag = torch.diagonal_scatter(mask, diag, dim1=-2, dim2=-1)
        adj_mat.masked_fill_(mask_ignore_diag.logical_not(), 0.0) 

    # # Visualise
    # plot_adj_mat(adj_mat, 0, 0)
    # plot_adj_mat(adj_mat, 0, 1)

    # Spatial Consistency Score
    lead_eigvec = batched_power_iteration(adj_mat)

    # # Visualise
    # plot_eigvec(lead_eigvec, 0, 0)
    # plot_eigvec(lead_eigvec, 0, 1)
    
    if not return_spatial_consistency:
        return lead_eigvec

    sc_score_list = batched_cal_spatial_consistency(adj_mat, lead_eigvec)
    sc_score_list = np.squeeze(sc_score_list.cpu().detach().numpy())
    return lead_eigvec, sc_score_list


def plot_adj_mat(adj_mat: Tensor, batch_idx=0, nn_idx=0):
    import matplotlib.pyplot as plt
    adj_mat = adj_mat.detach().cpu().numpy()
    if adj_mat.ndim == 3:
        adj_mat = adj_mat[nn_idx, ...]
    elif adj_mat.ndim == 4:
        adj_mat = adj_mat[batch_idx, nn_idx, ...]
    else:
        raise ValueError
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(adj_mat)

def plot_eigvec(eigvec: Tensor, batch_idx=0, nn_idx=0):
    import matplotlib.pyplot as plt
    eigvec = eigvec.detach().cpu().numpy()
    if eigvec.ndim == 2:
        eigvec = np.repeat(eigvec[nn_idx, ..., None], 10, axis=-1)
    elif eigvec.ndim == 3:
        eigvec = np.repeat(eigvec[batch_idx, nn_idx, ..., None], 10, axis=-1)
    else:
        raise ValueError
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xticks([])
    ax.imshow(eigvec)