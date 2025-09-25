"""
EgoNN loss functions, unified into one module  
Adapted from https://github.com/jac99/Egonn/
"""
from typing import List
import math

import numpy as np
import torch
import torch.nn as nn

from misc.poses import apply_transform
from models.losses.loss_utils import metrics_mean

EPS = 1e-5


class KeypointCorrLoss:
    # Loss combining keypoint loss and correspondence matrix loss
    def __init__(self, gamma_c=1., gamma_k=1., gamma_chamfer=1., gamma_p2p=1., beta=1., dist_th=0.5):
        # alpha and beta are parameters in Eq. (5) in RPM-Net paper
        self.keypoint_loss = KeypointLoss(gamma_chamfer=gamma_chamfer, gamma_p2p=gamma_p2p, prob_chamfer_loss=True,
                                          p2p_loss=True, repeatability_dist_th=dist_th)
        self.correspondence_loss = CorrespondenceLoss(beta=beta, dist_th=dist_th)

        self.gamma_k = gamma_k
        self.gamma_c = gamma_c

    def __call__(self, clouds1: List[torch.Tensor], keypoints1: torch.Tensor, sigma1: torch.Tensor, descriptors1: torch.Tensor,
                 clouds2: List[torch.Tensor], keypoints2: torch.Tensor, sigma2: torch.Tensor, descriptors2: torch.Tensor,
                 M_gt, *args, **kwargs):
        assert clouds1[0].dim() == 2
        assert clouds2[0].dim() == 2
        assert len(keypoints1) == len(sigma1)
        assert len(keypoints2) == len(sigma2)
        assert len(keypoints1) == len(descriptors1)
        assert len(keypoints2) == len(descriptors2)

        batch_metrics = []
        batch_loss = []

        for i, (kp1_pos, s1, desc1, kp2_pos, s2, desc2, Mi_gt) in enumerate(zip(keypoints1, sigma1, descriptors1,
                                                                                keypoints2, sigma2, descriptors2,
                                                                                M_gt)):
            pc1 = clouds1[i]
            pc2 = clouds2[i]

            kp1_pos_trans = apply_transform(kp1_pos, Mi_gt)
            dist_kp1_trans_kp2 = torch.cdist(kp1_pos_trans, kp2_pos)   # Euclidean distance

            metrics = {}
            metrics['kp_per_cloud'] = 0.5 * (len(kp1_pos) + len(kp2_pos))
            loss_keypoints, m = self.keypoint_loss(pc1, kp1_pos, s1, pc2, kp2_pos, s2, dist_kp1_trans_kp2)
            metrics.update(m)

            loss_correspondence, m = self.correspondence_loss(desc1, desc2, dist_kp1_trans_kp2)

            metrics.update(m)
            loss = self.gamma_k * loss_keypoints + self.gamma_c * loss_correspondence
            metrics['loss'] = loss.item()
            batch_metrics.append(metrics)
            batch_loss.append(loss)

        # Mean values per batch
        batch_loss = torch.stack(batch_loss).mean()
        batch_metrics = metrics_mean(batch_metrics)

        return batch_loss, batch_metrics


class KeypointLoss:
    # Computes loss for regressed keypoints (based on UNET paper)
    def __init__(self, gamma_chamfer=1., gamma_p2p=1., prob_chamfer_loss=True, p2p_loss=True,
                 repeatability_dist_th=0.5):
        # prob_chamfer_loss: if True probabilistic chamfer loss is calculated using predicted saliency uncertainty
        # p2p_loss: if True, point-2-point loss is calculated and added to chamfer loss
        self.gamma_chamfer = gamma_chamfer
        self.gamma_p2p = gamma_p2p
        self.prob_chamfer_loss = prob_chamfer_loss
        self.p2p_loss = p2p_loss
        self.repeatability_dist_th = repeatability_dist_th

    def __call__(self, pc1: torch.Tensor, kp1: torch.Tensor, sigma1: torch.Tensor,
                 pc2: torch.Tensor, kp2: torch.Tensor, sigma2: torch.Tensor,
                 dist_kp1_trans_kp2: torch.Tensor):
        # dist_kp1_trans_kp2: Euclidean distance matrix between transformed points in the first cloud
        #                     and points in the second cloud
        assert pc1.shape[1] == 3
        assert pc2.shape[1] == 3
        assert kp1.shape[1] == 3
        assert kp2.shape[1] == 3
        assert sigma1.shape[1] == 1
        assert sigma2.shape[1] == 1
        assert kp1.shape[0] == sigma1.shape[0]
        assert kp2.shape[0] == sigma2.shape[0]
        assert dist_kp1_trans_kp2.shape[0] == kp1.shape[0]
        assert dist_kp1_trans_kp2.shape[1] == kp2.shape[0]

        # Convert (n, 1) tensors to (n,) tensor
        sigma1 = sigma1.squeeze(1)
        sigma2 = sigma2.squeeze(1)

        min_dist1, min_dist_ndx1 = torch.min(dist_kp1_trans_kp2, dim=1)
        min_dist2, min_dist_ndx2 = torch.min(dist_kp1_trans_kp2, dim=0)

        # COMPUTE CHAMFER LOSS BETWEEN KEYPOINTS REGRESSED IN THE SOURCE AND TARGET KEYPOINT
        # In USIP code they use non-squared chamfer distance, in USIP paper squared distance is used

        # Match points from the first cloud to the closest points in the second cloud
        if self.prob_chamfer_loss:
            selected_sigma1 = sigma2[min_dist_ndx1]  # (n_keypoints1) tensor
            sigma12 = (sigma1 + selected_sigma1) / 2
            loss1 = (torch.log(sigma12) + min_dist1 / sigma12).mean()
        else:
            loss1 = min_dist1.mean()

        # Match points from the second cloud to the closest points in the first cloud
        if self.prob_chamfer_loss:
            selected_sigma2 = sigma1[min_dist_ndx2]  # (n_keypoints2) tensor
            sigma21 = (sigma2 + selected_sigma2) / 2
            loss2 = (torch.log(sigma21) + min_dist2 / sigma21).mean()
        else:
            loss2 = min_dist2.mean()

        # Metrics not included in optimization, for reporting only
        metrics = {}
        metrics['repeatability'] = torch.mean((min_dist1 <= self.repeatability_dist_th).float()).item()
        metrics['chamfer_pure'] = 0.5 * (min_dist1.detach().mean() + min_dist2.detach().mean()).item()
        if self.prob_chamfer_loss:
            weight_src_dst = (1.0 / sigma12.detach()) / (1.0 / sigma12.detach()).mean()
            weight_dst_src = (1.0 / sigma21.detach()) / (1.0 / sigma21.detach()).mean()
            chamfer_weighted = 0.5 * (weight_src_dst * min_dist1.detach()).mean() + \
                               0.5 * (weight_dst_src * min_dist2.detach()).mean()
            metrics['chamfer_weighted'] = chamfer_weighted.item()

        metrics['mean_sigma'] = 0.5 * (sigma12.detach().mean() + sigma21.detach().mean()).item()
        loss = self.gamma_chamfer * 0.5 * (loss1 + loss2)
        metrics['loss_chamfer'] = loss.item()

        if self.p2p_loss:
            # keypoints1 distance to the first point cloud
            dist1 = torch.cdist(kp1, pc1)
            min_dist1, _ = torch.min(dist1, dim=1)

            # keypoints2 distance to the first second point cloud
            dist2 = torch.cdist(kp2, pc2)
            min_dist2, _ = torch.min(dist2, dim=1)

            loss_p2p = 0.5 * (min_dist1.mean() + min_dist2.mean())
            metrics['loss_p2p'] = loss_p2p.item()
            loss = loss + self.gamma_p2p * loss_p2p

        metrics['keypoint_loss'] = loss.item()

        return loss, metrics


class CorrespondenceLoss(nn.Module):
    # Loss inspired by "Correspondence Matrices are Underrated" paper
    # Modified to handle partially overlapping clouds
    def __init__(self, beta, dist_th=0.5):
        # beta is not used currently
        super().__init__()
        self.beta = beta            # Inverse of the temperature
        self.dist_th = dist_th
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, desc1: torch.Tensor, desc2: torch.Tensor, dist_kp1_trans_kp2: torch.Tensor):
        # dist_kp1_trans_kp2: Euclidean distance matrix between transformed points in the first cloud
        #                     and points in the second cloud
        assert dist_kp1_trans_kp2.shape[0] == desc1.shape[0]
        assert dist_kp1_trans_kp2.shape[1] == desc2.shape[0]

        # For each keypoint from the first cloud, find it's true class, that is a corresponding keypoint from
        # the second cloud. Filter out keypoints without correspondence.
        min_dist1, min_dist_ndx1 = torch.min(dist_kp1_trans_kp2, dim=1)
        mask = min_dist1 <= self.dist_th

        similarity_matrix = torch.mm(desc1[mask], desc2.t()) * math.exp(self.beta)
        loss = self.loss(similarity_matrix, min_dist_ndx1[mask])
        # Keypoints having a corresponding keypoint in the other cloud (below dist_th meters)
        matching_keypoints = torch.sum(mask).float().item()
        # Keypoints that are correctly matched by searching for the closest descriptor
        if matching_keypoints > 0:
            tp_mask = torch.max(similarity_matrix, 1)[1] == min_dist_ndx1[mask]
            matching_descriptors = torch.sum(tp_mask).float().item()
            pos_similarity = torch.mean(torch.max(similarity_matrix, 1)[1].float()).item()
            neg_mat = similarity_matrix.clone().detach()
            neg_mat[:, min_dist_ndx1[mask]] = 0.
            neg_similarity = torch.mean(torch.max(neg_mat, 1)[0]).float().item()
        else:
            matching_descriptors = 0.
            pos_similarity = 0.
            neg_similarity = 0.

        metrics = {'correspondence_loss': loss.item(), 'matching_keypoints': matching_keypoints,
                   'matching_descriptors': matching_descriptors, 'pos_similarity': pos_similarity,
                   'neg_similarity': neg_similarity}
        return loss, metrics

