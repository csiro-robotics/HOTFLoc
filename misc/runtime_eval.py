"""
Evaluate runtime of MinkLoc and OctFormerLoc.

By Ethan Griffiths (ethan.griffiths@data61.csiro.au)
"""
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity
import ocnn
from ocnn.octree import Points, Octree
import MinkowskiEngine as ME

from models.model_factory import model_factory
from misc.utils import TrainingParams

def get_dummy_input(params: TrainingParams, input_size: int):
    x = torch.rand(input_size, 3, dtype=torch.float32)
    if params.load_octree:
        x = torch.clamp(x, -1, 1)
        # Convert to ocnn Points object, then create Octree
        points = Points(x)
        octree = Octree(params.octree_depth, full_depth=2)
        octree.build_octree(points)
        # NOTE: EVEN FOR A SINGLE SAMPLE, MERGE OCTREES MUST BE CALLED
        octree = ocnn.octree.merge_octrees([octree])
        octree.construct_all_neigh()
        input = {'octree': octree}
    else:  # sparse tensor
        quantizer = params.model_params.quantizer
        coords = quantizer(x)[0]
        coords = ME.utils.batched_coordinates([coords])
        # Assign a dummy feature equal to 1 to each point
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        input = {'coords': coords, 'features': feats}
    return input

if __name__ == "__main__":
    device = torch.device('cuda')
    input_size = 30000
    repetitions = 100
    configs = {}
    model_configs = {}
    configs['minkloc'] = '../config/config_baseline.txt'
    model_configs['minkloc'] = '../models/minkloc3dv2.txt'
    # configs['octfloc'] = '../config/config_baseline_octf_depth9_lr1e-4_sched70_350.txt'
    configs['octfloc'] = '../config/config_AboveUnder_baseline_octf_depth7_lr5e-4_mesa1_rotation_180deg_mode2.txt'
    model_configs['octfloc'] = '../models/octformer_4stage_best_cfg.txt'
    # configs['octfloc1stage'] = '../config/config_baseline_octf_depth8_lr1e-4_sched80_350.txt'
    # model_configs['octfloc1stage'] = '../models/octformer_1stage_18blocks_2ds_cfg.txt'
    # configs['hotfloc'] = '../config/config_baseline_octf_depth9_lr1e-4_sched70_350.txt'
    # model_configs['hotfloc'] = '../models/hotformer_4stage_2-18-2-2_no-ctprop_no-layerscale_cfg.txt'
    # configs['hotfloc_ADaPE'] = '../config/config_baseline_octf_depth9_lr1e-4_sched70_350.txt'
    # model_configs['hotfloc_ADaPE'] = '../models/hotformer_4stage_2-18-2-2_ADaPE_no-ctprop_no-layerscale_cfg.txt'
    # configs['hotfloc_pyramid_GeM'] = '../config/config_baseline_octf_depth9_lr5e-4_sched100_180.txt'
    # model_configs['hotfloc_pyramid_GeM'] = '../models/hotformerloc_3level_10blocks_ADaPE_0.5rtprop_PyramidOctGeM_cfg.txt'
    # configs['hotfloc_pyramid_AttnPoolMixer'] = '../config/config_baseline_octf_depth9_lr5e-4_sched100_180.txt'
    configs['hotfloc_pyramid_AttnPoolMixer'] = '../config/config_AboveUnder_baseline_octf_depth7_lr5e-4_mesa1_rotation_180deg_mode2.txt'
    model_configs['hotfloc_pyramid_AttnPoolMixer'] = '../models/hotformerloc_3level_10blocks_ADaPE_0.5rtprop_PyramidAttnPoolMixer_k74-36-18_cfg.txt'

    for model_type in configs.keys():
        print(f"EVALUATING {model_configs[model_type]} RUNTIME...")
        params = TrainingParams(configs[model_type], model_configs[model_type])
        model = model_factory(params.model_params).to(device)
        model.eval()

        # MEASURE DATA PRE-PROCESS TIME
        pre_process_timer = benchmark.Timer(
            stmt='get_dummy_input(params, input_size)',
            setup='from __main__ import get_dummy_input',
            globals={'params': params, 'input_size': input_size},
            label=f'Input pre-processing:',
        )
        print(pre_process_timer.timeit(repetitions))
        
        # INIT LOGGERS AND INPUT
        # TODO: GET A REAL POINT CLOUD SINCE RANDOM ONES PRODUCE DENSER OCTREES
        input = get_dummy_input(params, input_size)
        input = {e: input[e].to(device) for e in input}
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

        print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
        torch.cuda.reset_peak_memory_stats(device=None)