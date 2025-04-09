"""
Check positive threshold ranges for training tuple positives

Written by Ethan Griffiths (Data61, Pullenvale)
"""
import argparse 
import os
import pickle
import numpy as np
from tqdm import tqdm

def main():
    with open(args.training_tuples_path, 'rb') as f:
        train_tuples = pickle.load(f)

    min_dist_list = []
    max_dist_list = []
    for i in tqdm(range(len(train_tuples))):
        # Select anchor and positive
        anchor_path = os.path.join(args.dataset_root,
                                   train_tuples[i].rel_scan_filepath)
        anchor_position = train_tuples[i].position
        positive_path = None
        min_dist = np.nan
        max_dist = np.nan
        if args.ground_aerial and 'ground' not in anchor_path:
            continue
        for positive_id in train_tuples[i].positives:
            if args.ground_aerial and 'ground' in train_tuples[positive_id].rel_scan_filepath:
                continue
            positive_path = os.path.join(args.dataset_root,
                                            train_tuples[positive_id].rel_scan_filepath)
            positive_position = train_tuples[positive_id].position
            dist = np.linalg.norm(abs(anchor_position - positive_position))
            if dist < min_dist or np.isnan(min_dist):
                min_dist = dist
            if dist > max_dist or np.isnan(max_dist):
                max_dist = dist
        min_dist_list.append(min_dist)
        max_dist_list.append(max_dist)
            
    print(f"Min dist (min,avg,max): {np.nanmin(min_dist_list):.2f}, {np.nanmean(min_dist_list):.2f}, {np.nanmax(min_dist_list):.2f}")
    print(f"Max dist (min,avg,max): {np.nanmin(max_dist_list):.2f}, {np.nanmean(max_dist_list):.2f}, {np.nanmax(max_dist_list):.2f}")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type = str, required=True, help="root path of dataset")
    parser.add_argument('--training_tuples_path', type = str, required=True, help="path to training tuples pickle")
    parser.add_argument('--ground_aerial', action='store_true', help="Flag if measuring only ground/aerial pairs (CS-Campus3D, CS-Forest3D)")
    args = parser.parse_args()
    assert os.path.isdir(args.dataset_root), 'Invalid directory'
    assert os.path.isfile(args.training_tuples_path), 'Invalid path'
    main()