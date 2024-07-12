"""
Script to fix the broken timestamps recorded in poses_aligned.csv
"""
import os 
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KDTree 
import pickle 
import argparse 
from tqdm import tqdm 
from dataset.base_datasets import TrainingTuple
from dataset.WildPlaces.utils import load_csv, check_in_test_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--csv_filename', type=str, default='poses_aligned.csv', help='Name of CSV containing ground truth poses')
    parser.add_argument('--csv_savename', type=str, default='poses_aligned_fixed.csv', help='File to save fixed csv into')
    parser.add_argument('--cloud_folder', type=str, default='Clouds_downsampled', help='Name of folder containing point cloud frames')

    args = parser.parse_args()
    
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    print(f'Dataset root: {args.dataset_root}')

    ##### Venman #####
    base = os.path.join(args.dataset_root, 'Venman')
    train_folders = sorted(os.listdir(base))

    df_train_venman = pd.DataFrame(
        columns=['filename','northing','easting','x','y','z','qx','qy','qz','qw'])
    df_test_venman = pd.DataFrame(
        columns=['filename','northing','easting','x','y','z','qx','qy','qz','qw'])

    for folder in train_folders:
        df = pd.read_csv(os.path.join(base, folder, args.csv_filename), delimiter = ',', dtype = str)
        df = df.astype({'x': float, 'y':float, 'z':float, 'qx':float, 'qy':float, 'qz':float, 'qw':float, 'timestamp': str})
        df_fixed = df.copy()
        # Get list of all point cloud timestamps
        cloud_list = sorted(os.listdir(os.path.join(base, folder, args.cloud_folder)))
        correct_timestamp_list = [os.path.splitext(file)[0] for file in cloud_list]
        # store dict mapping timestamp values to correct filename
        timestamp_dict = {float(v): v for v in correct_timestamp_list}
        timestamp_array = np.asarray(list(timestamp_dict.keys()))

        for idx, row in tqdm(df.iterrows()):
            timestamp = row.timestamp
            correct_timestamp = correct_timestamp_list[idx]
            if timestamp != correct_timestamp: # mismatch
                df_fixed.loc[idx].timestamp = correct_timestamp
            
        # save poses
        df_fixed.to_csv(os.path.join(base, folder, args.csv_savename), index = False)

    ##### Karawatha #####
    base = os.path.join(args.dataset_root, 'Karawatha')
    train_folders = sorted(os.listdir(base))

    df_train_karawatha = pd.DataFrame(
        columns=['filename','northing','easting','x','y','z','qx','qy','qz','qw'])
    df_test_karawatha = pd.DataFrame(
        columns=['filename','northing','easting','x','y','z','qx','qy','qz','qw'])

    for folder in train_folders:
        df = pd.read_csv(os.path.join(base, folder, args.csv_filename), delimiter = ',', dtype = str)
        df = df.astype({'x': float, 'y':float, 'z':float, 'qx':float, 'qy':float, 'qz':float, 'qw':float, 'timestamp': str})
        df_fixed = df.copy()
        # Get list of all point cloud timestamps
        cloud_list = sorted(os.listdir(os.path.join(base, folder, args.cloud_folder)))
        correct_timestamp_list = [os.path.splitext(file)[0] for file in cloud_list]
        # store dict mapping timestamp values to correct filename
        timestamp_dict = {float(v): v for v in correct_timestamp_list}
        timestamp_array = np.asarray(list(timestamp_dict.keys()))

        for idx, row in tqdm(df.iterrows()):
            timestamp = row.timestamp
            correct_timestamp = correct_timestamp_list[idx]
            if timestamp != correct_timestamp: # mismatch
                df_fixed.at[idx, 'timestamp'] = correct_timestamp
            
        # save poses
        df_fixed.to_csv(os.path.join(base, folder, args.csv_savename), index = False)

