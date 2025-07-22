"""
PyTorch util functions
"""
import random
from typing import Union

import numpy as np

import torch
from ocnn.octree import Octree


def release_cuda(x, to_numpy=False):
    """Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item, to_numpy) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item, to_numpy) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value, to_numpy) for key, value in x.items()}
    elif isinstance(x, Octree):
        x = x.cpu()
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu()
            if to_numpy:
                x = x.numpy()
    return x


def to_device(x, device: Union[torch.device, str], non_blocking=True):
    """Move all tensors to device."""
    if isinstance(x, list):
        x = [to_device(item, device, non_blocking) for item in x]
    elif isinstance(x, tuple):
        x = (to_device(item, device, non_blocking) for item in x)
    elif isinstance(x, dict):
        x = {key: to_device(value, device, non_blocking) for key, value in x.items()}
    elif isinstance(x, (torch.Tensor, Octree)):
        x = x.to(device=device, non_blocking=non_blocking)
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