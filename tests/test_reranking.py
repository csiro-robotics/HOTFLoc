"""
Test validity of reranking functions.
"""
import pytest

import numpy as np
import torch

from models.reranking_utils import sgv_parallel, batched_sgv_parallel

ATOL = 1e-6
B, NN, NPTS, D = 16, 1, 128, 64
dummy_kpts_ref = torch.randn(B,1,NPTS,3) * 10.0
dummy_kpts_tgt = torch.randn(B,NN,NPTS,3) * 10.0
dummy_feats_ref = torch.randn(B,1,NPTS,D)
dummy_feats_tgt = torch.randn(B,NN,NPTS,D)


### Testing Batched SGV ###
def test_batched_sgv():
    leading_eigvec_list = []
    scores_list = []
    for batch_idx in range(B):
        leading_eigvec, score = sgv_parallel(
            dummy_kpts_ref[batch_idx], dummy_kpts_tgt[batch_idx],
            dummy_feats_ref[batch_idx], dummy_feats_tgt[batch_idx],
            return_spatial_consistency=True,
        )
        leading_eigvec_list.append(leading_eigvec)
        scores_list.append(score)
    leading_eigvecs = torch.stack(leading_eigvec_list, dim=0)
    scores = np.stack(scores_list, axis=0)

    leading_eigvecs_batched, scores_batched = batched_sgv_parallel(
        dummy_kpts_ref, dummy_kpts_tgt, dummy_feats_ref, dummy_feats_tgt,
        return_spatial_consistency=True,
    )
    
    assert torch.all(torch.isclose(leading_eigvecs, leading_eigvecs_batched, atol=ATOL))
    assert np.all(np.isclose(scores, scores_batched, atol=ATOL))

if __name__ == '__main__':
    test_batched_sgv()