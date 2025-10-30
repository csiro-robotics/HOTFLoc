# Test sets for Mulran dataset.
# Adapted from https://github.com/csiro-robotics/SpectralGV/blob/main/datasets/mulran/generate_evaluation_sets.py

import argparse
from typing import List, Tuple, Dict
import os
import pickle

from dataset.mulran.mulran_raw import MulranSequence
from dataset.base_datasets import EvaluationTuple, EvaluationSet
from dataset.dataset_utils import filter_query_elements

DEBUG = False


def get_scans(sequence: MulranSequence) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    for ndx in range(len(sequence)):
        pose = sequence.poses[ndx]
        position = pose[:2, 3]
        # Add sequence name to rel path, so Sejong and DCC queries will work with the same dataset_root
        item = EvaluationTuple(
            sequence.timestamps[ndx],
            os.path.join(args.sequence, sequence.rel_scan_filepath[ndx]),
            position=position,
            pose=pose,
        )
        elems.append(item)
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence: str, query_sequence: str, min_displacement: float = 0.2,
                            dist_threshold=20) -> EvaluationSet:
    split = 'test'
    map_sequence = MulranSequence(dataset_root, map_sequence, split=split, min_displacement=min_displacement)
    query_sequence = MulranSequence(dataset_root, query_sequence, split=split, min_displacement=min_displacement)

    map_set = get_scans(map_sequence)
    query_set = get_scans(query_sequence)

    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for Mulran dataset')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--sequence', type=str, required=False, default='Sejong')
    parser.add_argument('--min_displacement', type=float, default=0.2)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=20)
    args = parser.parse_args()

    print(f'Dataset root: {args.dataset_root}')
    print(f'Save dir: {args.save_dir}')
    print(f'Sequence: {args.sequence}')
    print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    # Sequences is a list of (map sequence, query sequence)
    if args.sequence == 'Sejong':
        sequences = [('Sejong_01', 'Sejong_02')]
    elif args.sequence == 'DCC':
        sequences = [('DCC_01', 'DCC_02')]
        args.min_displacement = 10.0
        args.dist_threshold = 5
    else:
        raise NotImplementedError
    if DEBUG:
        sequences = [('ParkingLot', 'ParkingLot')]

    for map_sequence, query_sequence in sequences:
        print(f'Map sequence: {map_sequence}')
        print(f'Query sequence: {query_sequence}')

        test_set = generate_evaluation_set(args.dataset_root, map_sequence, query_sequence,
                                           min_displacement=args.min_displacement, dist_threshold=args.dist_threshold)

        pickle_name = f'test_{map_sequence}_{query_sequence}_{args.min_displacement}_{args.dist_threshold}.pickle'
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            file_path_name = os.path.join(args.save_dir, pickle_name)
        else:
            file_path_name = os.path.join(args.dataset_root, pickle_name)
        test_set.save(file_path_name)