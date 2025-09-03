# Warsaw University of Technology

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance
from misc.utils import TrainingParams
from models.losses.truncated_smoothap import TruncatedSmoothAP
from models.losses.geotransformer_loss import OverallLoss


def make_losses(params: TrainingParams):
    if params.loss == 'batchhardtripletmarginloss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        global_loss_fn = BatchHardTripletLossWithMasks(params.margin)
    elif params.loss == 'batchhardcontrastiveloss':
        global_loss_fn = BatchHardContrastiveLossWithMasks(params.pos_margin, params.neg_margin)
    elif params.loss == 'truncatedsmoothap':
        global_loss_fn = TruncatedSmoothAP(tau1=params.tau1, similarity=params.similarity,
                                           positives_per_query=params.positives_per_query)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError

    # Loss applied to QKV projections to prevent collapse to zero
    if params.local_qkv_std_coeff > 0 or params.rt_qkv_std_coeff > 0:
        if params.rt_qkv_std_coeff > 0 and params.model_params.disable_rt:
            print('[WARNING] Value specified for relay token QKV std loss, but relay tokens are disabled')
            params.rt_qkv_std_coeff = 0  # set to zero to prevent key error in the loss func
        qkv_loss_fn = QKV_STD_Loss(
            local_qkv_std_coeff=params.local_qkv_std_coeff,
            rt_qkv_std_coeff=params.rt_qkv_std_coeff,
            qkv_target_std=params.qkv_target_std,
        )
    elif params.qkv_weight_norm_coeff > 0:
        qkv_loss_fn = QKV_Weight_Norm_Loss(
            qkv_weight_norm_coeff=params.qkv_weight_norm_coeff,
            qkv_target_norm=params.qkv_target_norm,
        )
    else:
        qkv_loss_fn = None

    rerank_loss_fn = None
    if params.rerank_loss_fn == 'batchhardbceloss':
        rerank_loss_fn = BatchHardRerankingBCELossWithMasks(
            batch_size=params.rerank_batch_size,
            loss_coeff=params.rerank_loss_coeff,
        )
    elif params.rerank_loss_fn is not None:
        raise NotImplementedError(f'Unknown re-ranking loss: {params.rerank_loss_fn}')

    local_loss_fn = None
    if params.local.enable_local:
        local_loss_fn = OverallLoss(params)

    return global_loss_fn, qkv_loss_fn, rerank_loss_fn, local_loss_fn


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance, max_triplets: Optional[int] = None):
        self.distance = distance
        self.max_triplets = max_triplets
        if self.max_triplets is not None and self.max_triplets <= 0:
            raise ValueError('Must sample at least 1 triplet')
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
        # Return top-k hardest negatives if batch size limited
        if self.max_triplets is not None and self.max_triplets < len(embeddings):
            _, topk_indices = torch.topk(hardest_negative_dist, k=self.max_triplets, largest=False)
            a = a[topk_indices]
            p = p[topk_indices]
            n = n[topk_indices]
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
        try:  # try and get the correct attribute from PML (num_past_filter is correct for versions somewhere between 1.1.2 <= version < 1.6.2, not exactly sure when it breaks)
            num_non_zero_triplets = self.loss_fn.reducer.num_past_filter
        except AttributeError:
            num_non_zero_triplets = self.loss_fn.reducer.triplets_past_filter

        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': num_non_zero_triplets,
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


class BatchHardRerankingBCELossWithMasks:
    """
    Re-ranking loss with BCE. Samples batch-hard triplets.
    """
    def __init__(self, batch_size: int, loss_coeff: float = 1.0):
        self.batch_size = batch_size
        assert self.batch_size >= 1
        self.loss_coeff = loss_coeff
        assert self.loss_coeff >= 0.0
        # Euclidean distance
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance, max_triplets=self.batch_size)
        self.loss_fn = torch.nn.BCELoss()

    def __call__(self, rerank_scores: Tensor, targets: Tensor):
        loss = self.loss_fn(rerank_scores, targets) * self.loss_coeff

        with torch.no_grad():
            stats = {
                'loss_rerank': loss.item(),
                'rerank_pos_score': rerank_scores[:,0].mean().item(),
                'rerank_neg_score': rerank_scores[:,1].mean().item(),
            }
        return loss, stats

    def get_hard_triplets(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        return hard_triplets


class QKV_Base_Loss(ABC):
    """
    Base class for QKV losses. Supports losses that operate on model outputs
    or the model itself.
    """
    def __init__(self):
        super().__init__()
        self._input_type = None
        self._valid_input_types = ['model_out', 'model']  # Model output or model itself
    
    @property
    def input_type(self) -> str:
        return self._input_type
    
    @input_type.setter
    def input_type(self, value: str):
        if value not in self._valid_input_types:
            raise ValueError(f'Invalid input type, must be one of {self._valid_input_types}')
        self._input_type = str.lower(value)

    def __call__(self, model_out: dict, model: torch.nn.Module):
        if self.input_type == 'model_out':
            return self._model_output_forward(model_out)
        elif self.input_type == 'model':
            return self._model_forward(model)
        else:
            raise NotImplementedError()

    @abstractmethod
    def _model_output_forward(self, model_out: dict):
        """Define a forward function that expects model outputs as input"""
        pass

    @abstractmethod
    def _model_forward(self, model: torch.nn.Module):
        """Define a forward function that expects model itself as input"""
        pass


class QKV_STD_Loss(QKV_Base_Loss):
    """
    Compute the standard deviation regularisation loss across the batch
    dimension of QKV projections. Expected input is the output from the
    HOTFormerLoc model.
    """
    def __init__(self, local_qkv_std_coeff=0, rt_qkv_std_coeff=0, qkv_target_std=1.0):
        super().__init__()
        assert local_qkv_std_coeff > 0 or rt_qkv_std_coeff > 0, (
            "Must specify loss coefficient for either local or relay token layers"
        )
        assert qkv_target_std >= 0, "Target std must be positive"
        self.input_type = 'model_out'
        self.local_qkv_std_coeff = local_qkv_std_coeff
        self.rt_qkv_std_coeff = rt_qkv_std_coeff
        self.qkv_target_std = qkv_target_std
        self._invalid_model_error = "Invalid model, currently only HOTFormerLoc supports QKV STD loss"

    def _model_output_forward(self, model_out: dict) -> tuple[torch.Tensor, dict]:
        local_qkv_std_loss_list = []  # store loss per layer to average it correctly later 
        rt_qkv_std_loss_list = []
        local_qkv_std_loss = 0
        rt_qkv_std_loss = 0
        stats = {'local_qkv_std': {}}
        # Calculate loss separately for local layers and relay token layers
        if self.local_qkv_std_coeff > 0:
            assert 'octf_qkv_std' in model_out and 'hosa_qkv_std' in model_out, self._invalid_model_error
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
            assert 'rt_qkv_std' in model_out, self._invalid_model_error
            rt_qkv_std_temp = torch.stack(model_out['rt_qkv_std'])
            stats['rt_qkv_std'] = torch.mean(rt_qkv_std_temp).item()
            rt_qkv_std_loss_list.extend(self.row_wise_std_loss(rt_qkv_std_temp))
            rt_qkv_std_loss = self.rt_qkv_std_coeff * torch.mean(torch.stack(rt_qkv_std_loss_list))
            stats['rt_qkv_std_loss'] = rt_qkv_std_loss.item()
        loss = local_qkv_std_loss + rt_qkv_std_loss
        return loss, stats

    def _model_forward(self):
        pass

    def row_wise_std_loss(self, std: torch.Tensor) -> torch.Tensor:
        """
        Computes the row-wise standard deviation regularisation loss. Takes a
        (N,d) tensor and returns an (N,) tensor with mean loss for each row.
        """
        if not std.ndim == 2:
            raise ValueError("Invalid tensor, expects tensor shape (N,d)")
        return torch.mean(F.relu(self.qkv_target_std - std), dim=-1)


class QKV_Weight_Norm_Loss(QKV_Base_Loss):
    """
    Loss applied to norm of QKV projection weights, penalising weight collapse
    to zero.
    """
    def __init__(self, qkv_weight_norm_coeff=0, qkv_target_norm=1.0):
        super().__init__()
        assert qkv_weight_norm_coeff > 0, "Must specify valid loss coefficient"
        assert qkv_target_norm > 0, "Must specify valid target norm"
        self.input_type = 'model'
        self.qkv_weight_norm_coeff = qkv_weight_norm_coeff 
        self.qkv_target_norm = qkv_target_norm
        self._invalid_model_error = "Invalid model, currently only OctFormer and HOTFormerLoc support QKV Weight Norm loss"

    def _model_forward(self, model: torch.nn.Module) -> tuple[torch.Tensor, dict]:
        weight_norm_list = []
        for name, layer in model.named_modules():
            if 'qkv' in str.lower(name) and isinstance(layer, torch.nn.Linear):
                weight_norm_list.append(layer.weight.norm(p='fro'))
        # NOTE: Initial weight norm is ~ >= 6, trained norm is < 0.15
        weight_norms = torch.stack(weight_norm_list)
        loss = self.shifted_log_weight_norm_loss(weight_norms)
        loss *= self.qkv_weight_norm_coeff
        stats = {'qkv_weight_norm_loss': loss.item(),
                 'qkv_weight_norm': torch.mean(weight_norms).item()}
        return loss, stats

    def _model_output_forward(self):
        pass

    def shifted_log_weight_norm_loss(self, weight_norms, epsilon=1e-8):
        """Penalises weight norm falling below qkv_target_norm."""
        penalty = -torch.log(weight_norms / self.qkv_target_norm + epsilon)
        penalty = F.relu(penalty)
        return torch.mean(penalty)


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