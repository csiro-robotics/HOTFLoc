"""
PyTorch util functions
"""
import random
from typing import Union

import numpy as np
import torch

from ocnn.octree import Octree, Points


def release_cuda(x, to_numpy=False):
    """Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item, to_numpy) for item in x]
    elif isinstance(x, tuple):
        x = tuple(release_cuda(item, to_numpy) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value, to_numpy) for key, value in x.items()}
    elif isinstance(x, Octree):
        x = x.cpu()
    elif isinstance(x, Points):
        batch_size = x.batch_size  # .to() method of Points is broken, doesn't transfer batch_size property
        x = x.cpu()
        x.batch_size = batch_size
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu()
            if to_numpy:
                x = x.numpy()
    return x


def to_device(x, device: Union[torch.device, str], non_blocking=False,
              construct_octree_neigh=False):
    """Move all tensors to device."""
    if isinstance(x, list):
        x = [to_device(item, device, non_blocking, construct_octree_neigh) for item in x]
    elif isinstance(x, tuple):
        x = tuple(to_device(item, device, non_blocking, construct_octree_neigh) for item in x)
    elif isinstance(x, dict):
        x = {key: to_device(value, device, non_blocking, construct_octree_neigh) for key, value in x.items()}
    elif isinstance(x, (torch.Tensor, Points)):
        x = x.to(device=device, non_blocking=non_blocking)
    elif isinstance(x, Octree):
        x = x.to(device=device, non_blocking=non_blocking)
        if construct_octree_neigh:
            with torch.no_grad():
                x.construct_all_neigh()  # Ensure octree neighbours are pre-computed
    return x


def set_seed(seed: int = 42):
    """
    Enable (mostly) deterministic behaviour in PyTorch.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def debug_time_func(func, num_repetitions: int = 1000, inputs = (None,)):
    """Time a function's runtime with CUDA events"""
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = torch.zeros((num_repetitions, 1))
    # GPU WARMUP
    for _ in range(10):
        _ = func(*inputs)
    # MEASURE PERFORMANCE
    for rep in range(num_repetitions):
        starter.record()
        _ = func(*inputs)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

    mean_syn = torch.sum(timings) / num_repetitions
    std_syn = torch.std(timings)
    print(f"{func.__class__} runtime:")
    print(f"  mean - {mean_syn:.2f}ms, std - {std_syn:.2f}ms")


def min_max_normalize(x: torch.Tensor, eps: float = 1e-8):
    """
    Scale values of a (*, N) tensor to [0, 1] along the last dimension.

    Args:
        x: (*, N) tensor
        eps: small constant to avoid division by zero

    Returns:
        scaled: (*, N) tensor with values in [0, 1]
    """
    min_vals, _ = x.min(dim=-1, keepdim=True)   # (*, 1)
    max_vals, _ = x.max(dim=-1, keepdim=True)   # (*, 1)
    scaled = (x - min_vals) / (max_vals - min_vals + eps)
    return scaled