# Generate evaluation sets
# - Map point clouds are taken from MapData folder
# - Query point clouds are taken from TestData
# For each area (BaylandsToSeafood, ColumbiaPark, HighWay237, MathildaAVE, SanJoseDowntown, SunnyvaleBigloop) a
# separate evaluation set is crated. We do not match clouds from different areas.
# Adapted from https://github.com/jac99/Egonn/blob/main/datasets/southbay/generate_evaluation_sets.py

import argparse
import numpy as np
from typing import List, Tuple, Dict
import os
import pickle

from dataset.southbay.southbay_raw import SouthBayDataset
from dataset.base_datasets import EvaluationTuple, EvaluationSet
from dataset.dataset_utils import filter_query_elements


def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def get_scans(ds: SouthBayDataset, split: str, area: str, min_displacement: float = 0.1) -> List[EvaluationTuple]:
    elems = []
    for ndx in ds.location_ndx[split][area]:
        pose = ds.global_ndx[ndx].pose
        position = pose[0:2, 3]       # (x, y) position in global coordinate frame
        rel_scan_filepath = ds.global_ndx[ndx].rel_scan_filepath
        timestamp = ds.global_ndx[ndx].timestamp

        item = EvaluationTuple(timestamp, rel_scan_filepath, position=position, pose=pose)
        elems.append(item)

    print(f"{len(elems)} total elements in {split} split")

    # Filter-out elements leaving only 1 per grid cell with min_displacement size
    pos = np.zeros((len(elems), 2), dtype=np.float32)
    for ndx, e in enumerate(elems):
        pos[ndx] = e.position

    # Quantize x-y coordinates. Quantized coords start from 0
    pos = np.floor(pos / min_displacement)
    pos = pos.astype(int)
    _, unique_ndx = np.unique(pos, axis=0, return_index=True)

    # Leave only unique elements
    elems = [elems[i] for i in unique_ndx]
    print(f"{len(elems)} filtered elements in {split} split with grid cell size = {min_displacement}")

    return elems


# def generate_evaluation_set(ds: SouthBayDataset, area: str, min_displacement: float = 0.1, dist_threshold=5) -> \
#         EvaluationSet:
#     map_set = get_scans(ds, 'MapData', area, min_displacement)
#     query_set = get_scans(ds, 'TestData', area, min_displacement)
#     query_set = filter_query_elements(query_set, map_set, dist_threshold)
#     print(f'Area: {area} - {len(map_set)} database elements, {len(query_set)} query elements\n')
#     return EvaluationSet(query_set, map_set)


# NOTE: Using the old eval set format for now, until the entire pipeline is upgraded (requires changing format for Oxford and CS-Campus3D, or keeping eval scripts separate)
def generate_evaluation_set(ds: SouthBayDataset, area: str, min_displacement: float = 0.1, dist_threshold=5) -> \
        Tuple[List[Dict]]:
    map_set = get_scans(ds, 'MapData', area, min_displacement)
    query_set = get_scans(ds, 'TestData', area, min_displacement)
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'Area: {area} - {len(map_set)} database elements, {len(query_set)} query elements\n')

    # Revert to old dict format
    query_set_old_format = [{}]
    map_set_old_format = [{}]
    for ii, query in enumerate(query_set):
        query_set_old_format[0][ii] = {
            'query': query.rel_scan_filepath,
            'easting': query.position[0],
            'northing': query.position[1],
            'pose': query.pose,
        }
    for ii, db in enumerate(map_set):
        map_set_old_format[0][ii] = {
            'query': db.rel_scan_filepath,
            'easting': db.position[0],
            'northing': db.position[1],
            'pose': db.pose,
        }
    return query_set_old_format, map_set_old_format


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for Apollo SouthBay dataset')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save eval tuples')
    parser.add_argument('--min_displacement', type=float, default=1.0)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=5)

    args = parser.parse_args()
    print(f'Dataset root: {args.dataset_root}')
    print(f'Save dir: {args.save_dir}')
    print(f'Minimum displacement between scans in each set (map/query): {args.min_displacement}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    ds = SouthBayDataset(args.dataset_root)
    ds.print_info()

    min_displacement = args.min_displacement

    area = 'SunnyvaleBigloop'   # Evaluation area
    assert area in ds.location_ndx['TestData']

    # eval_set = generate_evaluation_set(ds, area, min_displacement=min_displacement,
    #                                    dist_threshold=args.dist_threshold)
    # pickle_name = f'test_{area}_{args.min_displacement}_{args.dist_threshold}.pickle'
    # file_path_name = os.path.join(args.dataset_root, pickle_name)
    # eval_set.save(file_path_name)

    # NOTE: Using the old eval set format for now, until the entire pipeline is upgraded (requires changing format for Oxford and CS-Campus3D, or keeping eval scripts separate)
    query_set, map_set = generate_evaluation_set(
        ds, area, min_displacement=min_displacement, dist_threshold=args.dist_threshold
    )
    query_pickle_name = f'test_{area}_{args.min_displacement}_{args.dist_threshold}_query.pickle'
    map_pickle_name = f'test_{area}_{args.min_displacement}_{args.dist_threshold}_database.pickle'
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        query_file_path = os.path.join(args.save_dir, query_pickle_name)
        map_file_path = os.path.join(args.save_dir, map_pickle_name)
    else:
        query_file_path = os.path.join(args.dataset_root, query_pickle_name)
        map_file_path = os.path.join(args.dataset_root, map_pickle_name)
    
    output_to_file(query_set, query_file_path)
    output_to_file(map_set, map_file_path)