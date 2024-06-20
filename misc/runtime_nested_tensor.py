"""
Evaluate runtime of MinkLoc and OctFormerLoc.

By Ethan Griffiths (ethan.griffiths@data61.csiro.au)
"""
import random
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity
from ocnn.octree import Points, Octree
import MinkowskiEngine as ME

from models.model_factory import model_factory
from misc.utils import TrainingParams

class DummyModel(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # self.qkv = torch.nn.Linear(dim, dim * 3, bias=True)
        self.q = torch.nn.Linear(dim, dim, bias=True)
        self.k = torch.nn.Linear(dim, dim, bias=True)
        self.v = torch.nn.Linear(dim, dim, bias=True)
        self.proj = torch.nn.Linear(dim, dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, data: torch.Tensor):
        B = data.size(0)
        H = self.num_heads
        C = self.dim

        # data (N, _, C) 
        
        # # qkv
        # qkv = self.qkv(data).reshape(B, -1, 3, H, C // H).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]      # (B, H, _, C')
        q = self.q(data).reshape(B, -1, H, C // H).transpose(1, 2)  # (B, H, _, C')
        k = self.k(data).reshape(B, -1, H, C // H).transpose(1, 2)
        v = self.v(data).reshape(B, -1, H, C // H).transpose(1, 2)
        q = q * self.scale

        # attn
        attn = q @ k.transpose(-2, -1)        # (B, H, _, _)
        attn = self.softmax(attn)
        # data = (attn @ v).transpose(1, 2).reshape(-1, C)
        data = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        # ffn
        data = self.proj(data)
        return data

def get_dummy_input(min_size: int, max_size: int, dim: int, batch_size: int, nested = False):
    x = []
    for _ in range(batch_size - 1):
        N = random.randint(min_size, max_size)
        x.append(torch.rand(N, dim, dtype=torch.float32))
    x.append(torch.rand(max_size, dim, dtype=torch.float32))  # ensure 1 element is max size
    if nested:
        x = torch.nested.nested_tensor(x)
        # x = x.to_padded_tensor(padding=0.0)
    else:
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
    # x = torch.rand(batch_size, min_size, dim, dtype=torch.float32)
    return x

if __name__ == "__main__":
    device = torch.device('cuda')
    print(f"Cuda: {torch.cuda.is_available()}")
    max_size = 300
    min_size = 50
    batch_size = 128
    dim = 256
    num_heads = dim // 16
    repetitions = 100
    model = DummyModel(dim, num_heads).to(device)
    model.eval()
    
    # INIT LOGGERS AND INPUT
    for nested in [False, True]:
        print(f"NESTED: {nested}")
        # MEASURE DATA PRE-PROCESS TIME
        pre_process_timer = benchmark.Timer(
            stmt='get_dummy_input(min_size, max_size, dim, batch_size, nested=nested)',
            setup='from __main__ import get_dummy_input',
            globals={'min_size': min_size, 'max_size': max_size, 'dim': dim, 'batch_size': batch_size, 'nested': nested},
            label=f'Input pre-processing:',
        )
        print(pre_process_timer.timeit(repetitions))
        input = get_dummy_input(min_size, max_size, dim, batch_size, nested=nested)
        input = input.to(device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings=np.zeros((repetitions, 1))
        with torch.inference_mode():
            # GPU WARMUP
            for _ in range(10):
                _ = model(input)
            # MEASURE PERFORMANCE
            for rep in range(repetitions):
                starter.record()
                _ = model(input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

            mean_syn = np.sum(timings) / repetitions
            std_syn = np.std(timings)
            print(f"Model runtime:")
            print(f"  mean - {mean_syn:.2f}ms, std - {std_syn:.2f}ms")
                
            
        # # Profile model
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     record_shapes=True, profile_memory=True
        # ) as prof:
        #     with record_function("model_inference"):
        #         model(input)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
