"""
Evaluate runtime of MinkLoc and OctFormerLoc. This time using actual pointclouds
from the datasets (as random ones don't accurate represent how
voxelisation/octree generation will typically occur).

By Ethan Griffiths (ethan.griffiths@data61.csiro.au)
"""
import numpy as np
import os
import pickle
import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity
import ocnn
from ocnn.octree import Points, Octree
import MinkowskiEngine as ME

from models.model_factory import model_factory
from misc.utils import TrainingParams
from eval.utils import get_query_database_splits
from dataset.pointnetvlad.pnv_raw import PNVPointCloudLoader
from dataset.AboveUnder.AboveUnder_raw import AboveUnderPointCloudLoader
from dataset.augmentation import Normalize
from dataset.coordinate_utils import CylindricalCoordinates

# def get_dummy_input(params: TrainingParams, input_size: int):
#     x = torch.rand(input_size, 3, dtype=torch.float32)
#     if params.load_octree:
#         x = torch.clamp(x, -1, 1)
#         # Convert to ocnn Points object, then create Octree
#         points = Points(x)
#         octree = Octree(params.octree_depth, full_depth=2)
#         octree.build_octree(points)
#         # NOTE: EVEN FOR A SINGLE SAMPLE, MERGE OCTREES MUST BE CALLED
#         octree = ocnn.octree.merge_octrees([octree])
#         octree.construct_all_neigh()
#         input = {'octree': octree}
#     else:  # sparse tensor
#         quantizer = params.model_params.quantizer
#         coords = quantizer(x)[0]
#         coords = ME.utils.batched_coordinates([coords])
#         # Assign a dummy feature equal to 1 to each point
#         feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
#         input = {'coords': coords, 'features': feats}
#     return input

def load_eval_sets(params):
    eval_database_files, eval_query_files = get_query_database_splits(params)
    assert len(eval_database_files) == len(eval_query_files)

    # Just use default split (Oxford - Oxford, AboveUnder - Karawatha)
    database_file = eval_database_files[0]
    query_file = eval_query_files[0]
    
    # Extract location name from query and database files
    if 'AboveUnder' in params.dataset_name or 'CSWildPlaces' in params.dataset_name:
        if "pickles/" in database_file:  # CS-WildPlaces
            location_name = database_file.split('/')[-1].split('_')[1]
            temp = query_file.split('/')[-1].split('_')[1]
        else:
            location_name = database_file.split('_')[1]
            temp = query_file.split('_')[1]
    else:
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        if "5m-pickles/" in database_file:  # wild-places
            location_name = location_name.split('/')[-1]
            temp = temp.split('/')[-1]
    assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                   query_file)
    p = os.path.join(params.dataset_folder, database_file)
    with open(p, 'rb') as f:
        database_sets = pickle.load(f)

    p = os.path.join(params.dataset_folder, query_file)
    with open(p, 'rb') as f:
        query_sets = pickle.load(f)
    return query_sets, database_sets

def collate_batch(data, params: TrainingParams):
    if params.load_octree:
        octrees = ocnn.octree.merge_octrees(data)
        # NOTE: remember to construct the neighbor indices
        octrees.construct_all_neigh()
        # batch = {'octree': octrees.to(device)}
        batch = {'octree': octrees}
    else:
        coords = [params.model_params.quantizer(e)[0] for e in data]
        bcoords = ME.utils.batched_coordinates(coords)
        # Assign a dummy feature equal to 1 to each point
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        # batch = {'coords': bcoords.to(device), 'features': feats.to(device)}
        batch = {'coords': bcoords, 'features': feats}
    return batch

def load_preprocess_submap(submap, params, preprocess = True):
    """Load submap, and pre-process into appropriate format."""
    # Get correct point cloud loader
    if params.dataset_name in ['Oxford','CSCampus3D']:
        pc_loader = PNVPointCloudLoader()
    elif 'AboveUnder' in params.dataset_name or 'WildPlaces' in params.dataset_name:
        pc_loader = AboveUnderPointCloudLoader()
        
    pc_file_path = os.path.join(params.dataset_folder, submap["query"])
    data = pc_loader(pc_file_path)
    data = torch.tensor(data)
    print(f"Submap size: {len(data)}")
    if preprocess:
        data = preprocess_submap(data, params)
    return data

def preprocess_submap(data: torch.Tensor, params: TrainingParams):    
    if params.normalize_points or params.scale_factor is not None:
        normalize_transform = Normalize(scale_factor=params.scale_factor,
                                        unit_sphere_norm=params.unit_sphere_norm)

    if params.load_octree and params.model_params.coordinates == 'cylindrical':
        coord_converter = CylindricalCoordinates(use_octree=True)

    if params.normalize_points or params.scale_factor is not None:
        data = normalize_transform(data)
    if params.load_octree:  # Convert to Octree format
        # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
        mask = torch.all(abs(data) <= 1.0, dim=1)
        data = data[mask]
        # Also ensure this will hold if converting coordinate systems
        if params.model_params.coordinates == 'cylindrical':
            data_norm = torch.linalg.norm(data[:, :2], dim=1)[:, None]
            mask = torch.all(data_norm <= 1.0, dim=1)
            data = data[mask]
            # Convert to cylindrical coords
            data = coord_converter(data)
        # Convert to ocnn Points object, then create Octree
        points = Points(data)
        data = Octree(params.octree_depth, full_depth=2)
        data.build_octree(points)
    
    data = collate_batch([data], params)
    return data
    
if __name__ == "__main__":
    SET_IDX = -1
    # QUERY_IDX = 0
    QUERY_IDX = 1000 # CS-Wild-Places
    device = torch.device('cuda')
    repetitions = 300
    configs = {}
    model_configs = {}
    # configs['minkloc'] = '../config/config_baseline.txt'
    # configs['minkloc'] = '../config/config_Campus3D.txt'
    configs['minkloc'] = '../config/config_CS-WildPlaces_rmground_thresh30m_stdnorm.txt'
    model_configs['minkloc'] = '../models/minkloc3dv2.txt'
    
    # configs['octfloc'] = '../config/config_baseline_octf_depth9_lr1e-4_sched70_350.txt'
    # configs['octfloc'] = '../config/config_AboveUnder_baseline_octf_depth7_lr5e-4_mesa1_rotation_180deg_mode2.txt'
    # model_configs['octfloc'] = '../models/octformer_4stage_best_cfg.txt'
    # configs['octfloc1stage'] = '../config/config_baseline_octf_depth8_lr1e-4_sched80_350.txt'
    # model_configs['octfloc1stage'] = '../models/octformer_1stage_18blocks_2ds_cfg.txt'
    # configs['hotfloc'] = '../config/config_baseline_octf_depth9_lr1e-4_sched70_350.txt'
    # model_configs['hotfloc'] = '../models/hotformer_4stage_2-18-2-2_no-ctprop_no-layerscale_cfg.txt'
    # configs['hotfloc_ADaPE'] = '../config/config_baseline_octf_depth9_lr1e-4_sched70_350.txt'
    # model_configs['hotfloc_ADaPE'] = '../models/hotformer_4stage_2-18-2-2_ADaPE_no-ctprop_no-layerscale_cfg.txt'
    # configs['hotfloc_pyramid_GeM'] = '../config/config_baseline_octf_depth9_lr5e-4_sched100_180.txt'
    # model_configs['hotfloc_pyramid_GeM'] = '../models/hotformerloc_3level_10blocks_ADaPE_0.5rtprop_PyramidOctGeM_cfg.txt'
    # configs['hotfloc_pyramid_AttnPoolMixer'] = '../config/config_baseline_octf_depth9_lr5e-4_sched100_180.txt'

    # configs['hotfloc_Oxford'] = '../config/config_baseline_octf_depth9_lr5e-4_sched100_180.txt'
    # model_configs['hotfloc_Oxford'] = '../models/hotformerloc_best_oxford_cfg.txt'
    # configs['hotfloc_campus3d'] = '../config/config_Campus3D_octf_no-norm_depth7_lr5e-4_sched250_350_rotation_180deg_mode2.txt'
    # model_configs['hotfloc_campus3d'] = '../models/hotformerloc_best_CS-Campus3D_cfg.txt'
    configs['hotfloc_cswildplaces'] = '../config/config_hotformerloc_cs-wild-places.txt'
    model_configs['hotfloc_cswildplaces'] = '../models/hotformerloc_best_CS-Wild-Places_cfg.txt'

    # configs['hotfloc_cswildplaces_tiny'] = '../config/config_hotformerloc_cs-wild-places.txt'
    # model_configs['hotfloc_cswildplaces_tiny'] = '../models/hotformerloc-tiny_cfg.txt'
    query_sets, database_sets = None, None
    
    for model_type in configs.keys():        
        print(f"EVALUATING {model_configs[model_type]} RUNTIME...")
        params = TrainingParams(configs[model_type], model_configs[model_type])
        model = model_factory(params.model_params).to(device)
        model.eval()
        
        # Get queries and database for desired split
        # if query_sets is None:
        query_sets, database_sets = load_eval_sets(params)
            # print(query_sets[SET_IDX].keys())
            # print(database_sets[SET_IDX])

        # MEASURE DATA PRE-PROCESS TIME
        # query = query_sets[SET_IDX][QUERY_IDX]
        query = database_sets[SET_IDX][QUERY_IDX]
        query_submap = load_preprocess_submap(query, params, preprocess=False)
        pre_process_timer = benchmark.Timer(
            stmt='preprocess_submap(query_submap, params)',
            setup='from __main__ import preprocess_submap',
            globals={'query_submap': query_submap, 'params': params},
            label=f'Input pre-processing:',
        )
        print(pre_process_timer.timeit(repetitions))
        
        # INIT LOGGERS AND INPUT
        # TODO: GET A REAL POINT CLOUD SINCE RANDOM ONES PRODUCE DENSER OCTREES
        # input = get_dummy_input(params, input_size)
        input = load_preprocess_submap(query, params)
        input = {e: input[e].to(device) for e in input}
        # TODO: process OctreeT here as well, instead of in model.
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings=np.zeros((repetitions, 1))
        with torch.inference_mode():
            # GPU WARMUP
            for _ in range(10):
                _ = model(input, global_only=True)
            # MEASURE PERFORMANCE
            for rep in range(repetitions):
                starter.record()
                _ = model(input, global_only=True)
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
        #         model(input, global_only=True)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print(f"max mem allocated {torch.cuda.max_memory_allocated(device=None) / (1024 ** 2):.2f} MB memory")
        # print(f"mem allocated {torch.cuda.memory_allocated(device=None) / (1024 ** 2):.2f} MB memory")
        torch.cuda.reset_peak_memory_stats(device=None)