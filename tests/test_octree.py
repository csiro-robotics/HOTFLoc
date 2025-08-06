"""
Test validity of custom Octree operations.
"""
import pytest
import random

import torch
from ocnn.octree import Octree, Points, merge_octrees, merge_points

from models.octree import OctreeT, get_octant_centroids_from_points
from misc.torch_utils import set_seed

RANDOM_SEED = 42
ATOL = 1e-6
NUM_ITERS = int(100)
B = 64
N_MIN = 500
N_MAX = 20000
DEPTH = 7
FULL_DEPTH = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

### Testing get_octant_centroids_from_points ###
def test_get_octant_centroids_from_points():
    for _ in range(NUM_ITERS):
        dummy_batch = {'octree': [], 'points': []}
        for batch_id in range(B):
            n_pts = random.randint(N_MIN, N_MAX)
            dummy_points_raw = torch.randn(n_pts, 3, device=DEVICE) / 3
            dummy_points_raw = torch.clamp(dummy_points_raw, min=-1.0, max=1.0)
            dummy_points_temp = Points(dummy_points_raw)
            dummy_batch['points'].append(dummy_points_temp)
            dummy_octree_temp = Octree(DEPTH, full_depth=FULL_DEPTH, device=DEVICE)
            dummy_octree_temp.build_octree(dummy_points_temp)
            dummy_batch['octree'].append(dummy_octree_temp)
        # Merge into batch
        dummy_points = merge_points(dummy_batch['points'])
        dummy_octree = merge_octrees(dummy_batch['octree'])
        # Test func
        for depth in range(FULL_DEPTH+1, DEPTH):
            centroids_temp = get_octant_centroids_from_points(
                dummy_points, depth=depth, quantizer=None
            )
            assert(len(centroids_temp) == len(dummy_octree.batch_id(depth, nempty=True)))
            # # Below will fail, due to `to_points()` method altering the original points
            # centroids_from_octree_temp = get_octant_centroids_from_points(
            #     dummy_octree.to_points(), depth=depth, quantizer=None
            # )
            # assert(len(centroids_from_octree_temp) == len(dummy_octree.batch_id(depth, nempty=True)))


if __name__ == '__main__':
    set_seed(RANDOM_SEED)
    test_get_octant_centroids_from_points()