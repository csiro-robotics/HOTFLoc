"""
Re-ranking losses

Ethan Griffiths (Data61, Pullenvale)
"""

import numpy as np
import torch

from models.losses.loss_utils import sigmoid, compute_aff


class RerankingBCELoss:
    def __init__(self, ):
        self.loss_fn = torch.nn.BCELoss()
        
    def __call__(self, embeddings, positives_mask, negatives_mask):
        stats = {}

        # TODO:
        stats['loss'] = loss.item()
        return loss, stats
