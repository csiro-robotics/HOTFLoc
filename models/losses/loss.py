# Warsaw University of Technology

import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance
from misc.utils import TrainingParams
from models.losses.truncated_smoothap import TruncatedSmoothAP


def make_losses(params: TrainingParams):
    if params.loss == 'batchhardtripletmarginloss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(params.margin)
    elif params.loss == 'batchhardcontrastiveloss':
        loss_fn = BatchHardContrastiveLossWithMasks(params.pos_margin, params.neg_margin)
    elif params.loss == 'truncatedsmoothap':
        loss_fn = TruncatedSmoothAP(tau1=params.tau1, similarity=params.similarity,
                                    positives_per_query=params.positives_per_query)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError
    if params.local_qkv_std_coeff > 0 or params.rt_qkv_std_coeff > 0:
        qkv_std_loss_fn = QKV_STD_Loss(
            local_qkv_std_coeff=params.local_qkv_std_coeff,
            rt_qkv_std_coeff=params.rt_qkv_std_coeff,
            target_std=params.target_std
        )
    else:
        qkv_std_loss_fn = None

    return loss_fn, qkv_std_loss_fn


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist[a_keep_idx]).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist[a_keep_idx]).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist[a_keep_idx]).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist[a_keep_idx]).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist[a_keep_idx]).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist[a_keep_idx]).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class BatchHardTripletLossWithMasks:
    def __init__(self, margin:float):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)

        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.num_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }
        return loss, stats


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin: float, neg_margin: float):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # We use contrastive loss with squared Euclidean distance
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin,
                                              distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'pos_pairs_above_threshold': self.loss_fn.reducer.reducers['pos_loss'].pos_pairs_above_threshold,
                 'neg_pairs_above_threshold': self.loss_fn.reducer.reducers['neg_loss'].neg_pairs_above_threshold,
                 'pos_loss': self.loss_fn.reducer.reducers['pos_loss'].pos_loss.item(),
                 'neg_loss': self.loss_fn.reducer.reducers['neg_loss'].neg_loss.item(),
                 'num_pairs': 2*len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats


class QKV_STD_Loss:
    """
    Compute the standard deviation regularisation loss across the batch
    dimension of QKV projections. Expected input is the output from the
    HOTFormerLoc model.
    """
    def __init__(self, local_qkv_std_coeff=0, rt_qkv_std_coeff=0, target_std=1.0):
        assert local_qkv_std_coeff > 0 or rt_qkv_std_coeff > 0, (
            "Must specify loss coefficient for either local or relay token layers"
        )
        assert target_std >= 0, "Target std must be positive"
        self.local_qkv_std_coeff = local_qkv_std_coeff
        self.rt_qkv_std_coeff = rt_qkv_std_coeff
        self.target_std = target_std
        self.invalid_model_error = "Invalid model, currently only HOTFormerLoc supports QKV STD loss"

    def __call__(self, model_out: dict) -> torch.Tensor:
        local_qkv_std_loss_list = []  # store loss per layer to average it correctly later 
        rt_qkv_std_loss_list = []
        local_qkv_std_loss = 0
        rt_qkv_std_loss = 0
        stats = {'local_qkv_std': {}}
        # Calculate loss separately for local layers and relay token layers
        if self.local_qkv_std_coeff > 0:
            assert 'octf_qkv_std' in model_out and 'hosa_qkv_std' in model_out, self.invalid_model_error
            # Compute average qkv std loss per depth, since channel size can change between stages
            for idx, octf_stage_qkv_std in enumerate(model_out['octf_qkv_std'].values()):
                octf_qkv_std_temp = torch.stack(octf_stage_qkv_std)
                stats['local_qkv_std'][idx] = torch.mean(octf_qkv_std_temp).item()
                local_qkv_std_loss_list.extend(self.row_wise_std_loss(octf_qkv_std_temp))
            for idx, hosa_stage_qkv_std in enumerate(model_out['hosa_qkv_std'].values()):
                hosa_qkv_std_temp = torch.stack(hosa_stage_qkv_std)
                stats['local_qkv_std'][len(model_out['octf_qkv_std']) + idx] = (
                    torch.mean(hosa_qkv_std_temp).item()
                )
                local_qkv_std_loss_list.extend(self.row_wise_std_loss(hosa_qkv_std_temp))
            local_qkv_std_loss = self.local_qkv_std_coeff * torch.mean(torch.stack(local_qkv_std_loss_list))
            stats['local_qkv_std_loss'] = local_qkv_std_loss.item()
        if self.rt_qkv_std_coeff > 0:
            assert 'rt_qkv_std' in model_out, self.invalid_model_error
            rt_qkv_std_temp = torch.stack(model_out['rt_qkv_std'])
            stats['rt_qkv_std'] = torch.mean(rt_qkv_std_temp).item()
            rt_qkv_std_loss_list.extend(self.row_wise_std_loss(rt_qkv_std_temp))
            rt_qkv_std_loss = self.rt_qkv_std_coeff * torch.mean(torch.stack(rt_qkv_std_loss_list))
            stats['rt_qkv_std_loss'] = rt_qkv_std_loss.item()
        loss = local_qkv_std_loss + rt_qkv_std_loss
        return loss, stats

    def row_wise_std_loss(self, std: torch.Tensor) -> torch.Tensor:
        """
        Computes the row-wise standard deviation regularisation loss. Takes a
        (N,d) tensor and returns an (N,) tensor with mean loss for each row.
        """
        if not std.ndim == 2:
            raise ValueError("Invalid tensor, expects tensor shape (N,d)")
        return torch.mean(F.relu(self.target_std - std), dim=-1)


def kdloss(y, teacher_scores):
    """
    Adapted from FasterViT repo:
    https://github.com/NVlabs/FasterViT/blob/main/fastervit/train.py#L356
    """
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()
    T = 3
    p = torch.nn.functional.log_softmax(y/T, dim=1)
    q = torch.nn.functional.softmax(teacher_scores/T, dim=1)
    l_kl = 50.0*kl_loss(p, q)
    return l_kl