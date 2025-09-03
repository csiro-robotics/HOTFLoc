"""
Generate train, test, and eval tuples for the CS-Wild-Places dataset. Generates 
baseline set of training queries, and test queries and database for all
environments. Stores training queries in v3 format, with pose and relative poses
of all positive pairs.

By Ethan Griffiths (Data61, Pullenvale)
"""
from os import path as osp
from os import listdir, makedirs
import pickle
import random
import math
import argparse
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.base_datasets import TrainingTuple, EvaluationTuple
from dataset.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader
from misc.point_clouds import icp, make_open3d_point_cloud
from misc.poses import relative_pose, xyz_quat2m
from misc.average_meter import AverageMeter

pd.options.mode.chained_assignment = None  # default='warn', disables warning
CLOUD_DIR = 'clouds/'
POSES_FILE = 'poses.csv'
DF_COLUMNS = ['file','easting','northing', 'x','y','z','qx','qy','qz','qw']

# Initialise random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
MARKER_SIZES = {'aerial':4, 'ground':8}

VAL_FOLDERS = ['2021_06_22_K-01_ground_sample0.5s', '2021_06_21_K-02_ground_sample0.5s',
               '2024_07_10_aerial_sample10m']   # splits to use for validation during training
BASELINE_SPLITS = ['Karawatha', 'Venman']  # splits in baseline train set

## NEW TEST REGIONS WHICH COVER THE ENTIRE SPLITS
# For QCAT
p1 = Polygon([(490500, 6955000), (490500, 6956000), (491500, 6956000), (491500, 6955000)])
# For Samford
p2 = Polygon([(487000, 6969000), (487000, 6971000), (489000, 6971000), (489000, 6969000)])

# For Karawatha (same as wild places, but transformed to UTM frame)
p6 = Polygon([(507018.60467,6942659.3756), (507468.60473,6942659.6724),
              (507468.74853,6942441.6724), (507018.74850,6942441.3756)])
p7 = Polygon([(506953.20227,6943269.3327), (507094.20227,6943269.4257),
              (507094.33093,6943074.4257), (506953.33090,6943074.3327)])
p8 = Polygon([(506655.41198,6942951.1361), (506655.58551,6942688.1361),
              (506847.58554,6942688.2628), (506847.41204,6942951.2627)])
# For Venman (same as wild places, but transformer to UTM frame)
p9 = Polygon([(519331.85162354,6943652.20440674), (519331.19000244,6943778.20266724),
              (519485.18786621,6943779.01129150), (519494.35580444,6943747.05899048),
              (519607.18621826,6943779.65188599), (519607.84783936,6943653.65362549)])
p10 = Polygon([(519722.31359863,6943565.25347900), (519722.54461670,6943521.25408936),
               (519495.54779053,6943520.06213379), (519495.31674194,6943564.06152344)])
p11 = Polygon([(519737.04788208,6943806.33413696), (519894.04573059,6943807.15850830),
               (519941.41265869,6943737.40628052), (519940.15832520,6943595.39773560),
               (519738.16110229,6943594.33709717)])

# Exclude Karawatha ground region with high drift
kara_exclude1 = Polygon([[507460, 6942602], [507930, 6942602], [507930, 6941940], 
                         [506990, 6941940], [506990, 6942340], [507460, 6942340]])

POLY_DICT = {'QCAT':[p1], 'Samford':[p2], 'Karawatha':[p6,p7,p8], 'Venman':[p9, p10, p11]}
GROUND_EXCLUDE_DICT = {'QCAT':[], 'Samford':[], 'Karawatha':[kara_exclude1], 'Venman':[]}
###

def get_timestamp_from_file(file):
    timestamp = str(osp.splitext(osp.split(file)[-1])[0])
    return timestamp

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

def check_in_test_set(easting, northing, test_polygons: List,
                      ground_exclude_polygons: List, run_type: str,
                      test_queries : KDTree = None):
    """
    Return which split submap is in, given polygon containing test queries. Run
    twice, first to get the ground test queries, and second to find all submaps
    in the buffer zone.
    """
    split = 'train'
    point = Point(easting, northing)
    for exclude_poly in ground_exclude_polygons:  # Only exclude from ground submaps
        if exclude_poly.contains(point) and run_type == 'ground':
            split = 'buffer'
            return split
    for test_poly in test_polygons:
        if test_poly.contains(point) and run_type == 'ground':
            split = 'test'
            return split
    if test_queries is not None:
        coord = np.array([easting, northing]).reshape(1, -1)
        num_matches = test_queries.query_radius(coord, args.buffer_thresh, count_only=True)
        if num_matches[0] > 0:
            split = 'buffer'
    return split

def cache_pcs(df_centroids: pd.DataFrame, voxel_size: Optional[float] = None) -> dict:
    """
    Pre-load all submaps in the train/test set into a dictionary, to speed up
    ICP computation. Dictionary contains numpy submaps idxed with same idx as
    df_centroids.
    """
    pc_dict = {}
    pc_loader = CSWildPlacesPointCloudLoader()
    for ndx, row in tqdm(df_centroids.iterrows(), total=len(df_centroids),
                         desc='Caching submaps for ICP'):
        pc = pc_loader(osp.join(args.root, row["file"]))
        if voxel_size is not None:
            pc_o3d = make_open3d_point_cloud(pc)
            pc_o3d = pc_o3d.voxel_down_sample(voxel_size=voxel_size)
            pc = np.asarray(pc_o3d.points, dtype=pc.dtype)
        pc_dict[ndx] = pc
    
    return pc_dict

def construct_training_query_dict(df_centroids, filename_base, test_set=False, icp_voxel_size=0.8):
    """
    Create training query dictionaries with v3 (EgoNN-esque) format.
    """
    run_str = 'test' if test_set else 'training'
    print(f"Computing {run_str} queries...")
    pc_loader = CSWildPlacesPointCloudLoader()
    pickle_save_file = filename_base + 'v3.pickle'
    tree = KDTree(df_centroids[['easting','northing']])
    ind_pos = tree.query_radius(
        df_centroids[['easting','northing']], r=args.pos_thresh
    )
    # store ground indices to remove from test set positives (ONLY EVAL WITH GROUND QUERY, AERIAL DATABASE)
    cloud_files = df_centroids['file'].to_numpy()
    ind_ground = np.array([i for i, x in enumerate(cloud_files) if 'ground' in x])
    ind_aerial = np.array([i for i, x in enumerate(cloud_files) if 'aerial' in x])
    ind_non_neg = tree.query_radius(
        df_centroids[['easting','northing']], r=args.neg_thresh
    )
    ind_df_centroids = df_centroids.index.values.tolist()
    queries_dict = {}
    num_queries_skipped = {
        split:0 for split in POLY_DICT.keys()
    }

    # Cache submaps for ICP computation (saves IO in the long run)
    if args.icp and args.cache_submaps:
        pc_dict = cache_pcs(df_centroids, voxel_size=icp_voxel_size)

    count_no_positives = dict.fromkeys(['ground','aerial'], 0)
    fitness_meter = AverageMeter()
    inlier_rmse_meter = AverageMeter()
    pbar = tqdm(range(len(ind_pos)), desc='Computing')
    for anchor_ndx in pbar:
        anchor_position = np.array(
            df_centroids.iloc[anchor_ndx][['easting', 'northing']],
            dtype=np.float64,
        )
        anchor_pose_xyz_quat = np.array(
            df_centroids.iloc[anchor_ndx][['x','y','z','qx','qy','qz','qw']],
            dtype=np.float64,
        )
        anchor_pose = xyz_quat2m(anchor_pose_xyz_quat)
        query = df_centroids.iloc[anchor_ndx]['file']
        split = str.split(query, '/')[0]    # first component is split
        # Extract timestamp from the filename
        timestamp = get_timestamp_from_file(query)
        
        positives = np.setdiff1d(ind_pos[anchor_ndx], [anchor_ndx])
        negatives = np.setdiff1d(ind_df_centroids, ind_non_neg[anchor_ndx])
        non_negatives = np.sort(ind_non_neg[anchor_ndx])

        # ICP pose refinement
        positives_poses = {}

        # remove queries with no ground positives, or remove all aerial queries if creating test set
        if (test_set and 'aerial' in query) or (
            args.query_requires_ground
            and 'aerial' in query
            and not any(
                ['ground' in file for file in df_centroids.iloc[positives]['file']]
            )
        ):
            num_queries_skipped[split] += 1
            # NOTE: Batch sampler now handles removing aerial queries
            positives = np.setdiff1d(positives, ind_aerial)
            negatives = np.setdiff1d(negatives, ind_aerial)
            non_negatives = np.union1d(non_negatives, ind_aerial)
        # remove ground positives/negatives from test set
        elif test_set and 'ground' in query: 
            positives = np.setdiff1d(positives, ind_ground)
            negatives = np.setdiff1d(negatives, ind_ground)
            non_negatives = np.union1d(non_negatives, ind_ground)

        # remove ground/ground and aerial/aerial positives
        if args.ground_aerial_positives_only:
            if 'ground' in query:
                positives = np.setdiff1d(positives, ind_ground)
                negatives = np.setdiff1d(negatives, ind_ground)
                non_negatives = np.union1d(non_negatives, ind_ground)
            elif 'aerial' in query:
                positives = np.setdiff1d(positives, ind_aerial)
                negatives = np.setdiff1d(negatives, ind_aerial)
                non_negatives = np.union1d(non_negatives, ind_aerial)
        np.random.shuffle(negatives)

        if len(positives) == 0:
            if 'ground' in query:
                count_no_positives['ground'] += 1
            elif 'aerial' in query:
                count_no_positives['aerial'] += 1
                
        if args.icp:
            if args.cache_submaps:
                anchor_pc = pc_dict[anchor_ndx]
            else:
                anchor_pc = pc_loader(
                    osp.join(args.root, df_centroids.iloc[anchor_ndx]["file"])
                )

        queries_dict[anchor_ndx] = TrainingTuple(
            id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
            positives=positives, non_negatives=non_negatives, 
            position=anchor_position, pose=anchor_pose,
        )
        if args.ignore_positives_poses:
            continue
        
        # tic = time.perf_counter()
        for positive_ndx in positives:
            positive_pose_xyz_quat = np.array(
                df_centroids.iloc[positive_ndx][['x','y','z','qx','qy','qz','qw']],
                dtype=np.float64,
            )
            positive_pose = xyz_quat2m(positive_pose_xyz_quat)
            # Compute initial relative pose
            transform = relative_pose(anchor_pose, positive_pose)
            positives_poses[positive_ndx] = transform
            if args.icp:
                # Refine the pose using ICP
                if args.cache_submaps:
                    positive_pc = pc_dict[positive_ndx]
                else:
                    positive_pc = pc_loader(
                        osp.join(args.root, df_centroids.iloc[positive_ndx]["file"])
                    )
                m, fitness, inlier_rmse = icp(
                    anchor_pc, positive_pc, transform,
                    voxel_size=(None if args.cache_submaps else icp_voxel_size),
                )
                fitness_meter.update(fitness)
                inlier_rmse_meter.update(inlier_rmse)
                positives_poses[positive_ndx] = m
                pbar.set_description(f"Fitness: {fitness_meter.mean():.4f} ({fitness_meter.std():.4f})"
                                     f" -- Inlier RMSE: {inlier_rmse_meter.mean():.4f} ({inlier_rmse_meter.std():.4f})")
        # icp_time = time.perf_counter() - tic
        # print(f"{len(positives)} positives took {icp_time:.1f} secs", flush=True) 
        queries_dict[anchor_ndx].positives_poses = positives_poses
    
    print(f"Queries with no positives: ground - {count_no_positives['ground']},"
          f" aerial - {count_no_positives['aerial']}")
    print("Queries skipped per split:")
    num_queries_skipped_total = 0
    for split, num in num_queries_skipped.items():
        print(f"{split}: {num}")
        num_queries_skipped_total += num
    if args.icp:
        # avg_fitness = np.mean(fitness_l)
        # avg_inlier_rmse = np.mean(inlier_rmse_l)
        avg_fitness = fitness_meter.mean()
        avg_inlier_rmse = inlier_rmse_meter.mean()
        print("ICP Results:")
        print(f"\tAvg fitness: {avg_fitness}")
        print(f"\tAvg inlier RMSE: {avg_inlier_rmse}")
    final_num_queries = len(queries_dict) - sum(count_no_positives.values())
    print(f"Final number of {run_str} queries: {final_num_queries}/{len(queries_dict)}")
    output_to_file(queries_dict, pickle_save_file)

def construct_query_and_database_sets(database_trees, database_sets, test_sets, filename_base):
    # TODO: Add option to compute positives by overlap, rather than distance threshold
    print("Saving queries and database...")
    eval_thresh = args.eval_thresh    
    file_db = filename_base + "_database.pickle"
    file_query = filename_base + "_query.pickle"
    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if(i == j):
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array(
                    [[test_sets[j][key]["easting"],
                      test_sets[j][key]["northing"]]])
                if (tree is None): # skip empty tree
                    test_sets[j][key][i] = []
                else:
                    index = tree.query_radius(coor, r=eval_thresh)
                    # indices of the positive matches in database i of each query (key) in test set j
                    test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, file_db)
    output_to_file(test_sets, file_query)
    
    return True

def format_df(df_locations: pd.DataFrame, clouds_relpath: str):
    # Fix column names (leave x and y for converting pose)
    df_locations.insert(1, 'easting', df_locations['x'])
    df_locations.insert(2, 'northing', df_locations['y'])
    df_locations = df_locations[['timestamp', *DF_COLUMNS[1:]]]
    
    # Create filepath from timestamp
    df_locations.loc[:,'timestamp'] = (clouds_relpath
                                       + df_locations.loc[:,'timestamp']
                                       + '.pcd')
    df_locations.rename(columns={'timestamp':'file'}, inplace=True)
    
    return df_locations

def main():
    root_dir = args.root
    save_dir = args.save_dir
    splits = args.splits
    
    if not osp.exists(save_dir):
        makedirs(save_dir)

    # Initialize pandas DataFrame
    df_train_baseline = pd.DataFrame(columns=DF_COLUMNS)
    df_test = pd.DataFrame(columns=DF_COLUMNS)
    test_queries = []

    # For visualisations
    all_coords = {}
    all_colours = {}
    all_sizes = {}

    # Find splits in dataset folder
    if len(splits) == 0:
        splits = sorted(listdir(root_dir))
        splits = [x for x in splits if osp.isdir(osp.join(root_dir, x))]
    for split in splits:
        print(f'Processing {split}')
        if split not in POLY_DICT.keys():
            warnings.warn('Split is not recognised, no test areas are '
                          'associated with it. Ignoring...')
            continue
        all_coords[split] = []
        all_colours[split] = []
        all_sizes[split] = []
        folders = sorted(listdir(osp.join(root_dir, split)))
        print(f'Folders:\n{folders}')

        # Check folders are valid
        for folder in folders:
            assert 'ground' in folder or 'aerial' in folder, \
            f'Invalid folder "{folder}", must contain aerial or ground in name'

        # Separate ground folders
        ground_folders = [folder for folder in folders if 'ground' in folder]
        
        # Determine ground test queries
        print('Getting ground queries... ', end='')
        for folder in ground_folders:
            run_type = 'ground'
            df_locations = pd.read_csv(
                osp.join(root_dir, split, folder, POSES_FILE), 
                sep=',', 
                dtype={'timestamp':str}
            )
            
            # Get easting and northing
            coords = df_locations[['x','y']].to_numpy()

            # Find ground queries
            for row in coords:
                row_split = check_in_test_set(row[0], row[1], POLY_DICT[split], 
                                              GROUND_EXCLUDE_DICT[split],
                                              run_type, None)
                if row_split == 'test':
                    test_queries.append(row)                
        
        print('Done')
        if len(test_queries) == 0:
            print(f'WARNING: No test queries found for {split}, all will be in training set')
            test_queries_tree = None
        else:
            test_queries_tree = KDTree(test_queries)
        
        # Reset counters
        test_counter = dict.fromkeys(['aerial','ground'], 0)
        buffer_counter = dict.fromkeys(['aerial','ground'], 0)
        train_counter = dict.fromkeys(['aerial','ground'], 0)

        database_trees = []
        database_sets = []
        test_sets = []

        # Gather submaps from each folder in split        
        print('Processing submaps... ', end='')
        for folder in folders:
            df_database = pd.DataFrame(columns=DF_COLUMNS)
            database_dict = {}
            test_dict = {}
            if 'aerial' in folder:
                run_type = 'aerial' 
            elif 'ground' in folder:
                run_type = 'ground'
            else:
                raise ValueError(f'Invalid folder "{folder}", '
                                  'must contain aerial or ground in name')
            
            df_locations = pd.read_csv(
                osp.join(root_dir, split, folder, POSES_FILE),
                sep=',',
                dtype={'timestamp':str}
            )
            
            # Fix column names and filenames
            clouds_relpath = osp.join(split, folder, CLOUD_DIR)
            df_locations = format_df(df_locations, clouds_relpath)
            
            # Sort submaps by train, test, and buffer set
            for _, row in tqdm(df_locations.iterrows(), desc=folder, 
                               total=len(df_locations)):
                assert osp.isfile(osp.join(root_dir, row['file'])), \
                    f"No associated submap for pose: {row['file']}"
                all_coords[split].append(row[['easting','northing']])
                all_sizes[split].append(MARKER_SIZES[run_type])
                row_pose = xyz_quat2m(
                        np.array(row[['x','y','z','qx','qy','qz','qw']], dtype=np.float64)
                )
                row_split = check_in_test_set(row['easting'], row['northing'], 
                                              POLY_DICT[split],
                                              GROUND_EXCLUDE_DICT[split],
                                              run_type, test_queries_tree)
                if row_split == 'test':
                    if folder in VAL_FOLDERS:  # test queries only consider certain splits, for consistency with other models (as minkloc3dv2 is the only to validate using the test query tuple)
                        df_test.loc[len(df_test)] = row
                    # test_tuple = EvaluationTuple(  # Can upgrade to EvaluationTuple class in future
                    #     timestamp=get_timestamp_from_file(row['file']),
                    #     rel_scan_filepath=row['file'],
                    #     pose=row_pose,
                    #     position=np.array(row[['easting','northing']], dtype=np.float64)
                    # )
                    # test_dict[len(test_dict.keys())] = test_tuple
                    test_dict[len(test_dict.keys())] = {
                        'query': row['file'],
                        'easting': row['easting'],
                        'northing':row['northing'],
                        'pose': row_pose,
                    }
                    test_counter[run_type] += 1
                    all_colours[split].append([1,0,0])
                elif row_split == 'buffer':
                    buffer_counter[run_type] += 1
                    all_colours[split].append([1,165/255,0])
                else:
                    if split in BASELINE_SPLITS:
                        df_train_baseline.loc[len(df_train_baseline)] = row
                    train_counter[run_type] += 1
                    all_colours[split].append([0,0,1])
                    
                if run_type == 'aerial':    # all aerial submaps form database
                    if folder in VAL_FOLDERS:
                        df_test.loc[len(df_test)] = row
                    df_database.loc[len(df_database)] = row
                    # database_tuple = EvaluationTuple(  # Can upgrade to EvaluationTuple class in future
                    #     timestamp=get_timestamp_from_file(row['file']),
                    #     rel_scan_filepath=row['file'],
                    #     pose=row_pose,
                    #     position=np.array(row[['easting','northing']], dtype=np.float64)
                    # )
                    # database_dict[len(database_dict.keys())] = database_tuple
                    database_dict[len(database_dict.keys())] = {
                        'query': row['file'],
                        'easting': row['easting'],
                        'northing': row['northing'],
                        'pose': row_pose,
                    }
            database_tree = KDTree(df_database[['easting','northing']]) if not df_database.empty else None
            database_trees.append(database_tree)
            database_sets.append(database_dict)
            test_sets.append(test_dict)

        print('Done')
        # save query/db pickles
        filename_base = osp.join(save_dir, f"CSWildPlaces_{split}_evaluation")
        construct_query_and_database_sets(database_trees, database_sets, test_sets, filename_base)

        len_database_sets = sum([len(database_set) for database_set in database_sets])
        len_test_sets = sum([len(test_set) for test_set in test_sets])
        print(f'{split} stats:\n'
            f'\tTraining submaps - {train_counter["aerial"] + train_counter["ground"]} '
            f'({train_counter["aerial"]} aerial, {train_counter["ground"]} ground)\n'
            f'\tTest submaps     - {test_counter["aerial"] + test_counter["ground"]} '
            f'({test_counter["aerial"]} aerial, {test_counter["ground"]} ground)\n'
            f'\tBuffer submaps   - {buffer_counter["aerial"] + buffer_counter["ground"]} '
            f'({buffer_counter["aerial"]} aerial, {buffer_counter["ground"]} ground)\n'
            f'  Eval ground queries  - {len_test_sets} possible\n'
            f'  Eval aerial database - {len_database_sets}'
        )
    print(f"\nTotal number of potential baseline training submaps: "
          f"{len(df_train_baseline['file'])}")
    print(f"Total number of test query submaps: {len(test_queries)}")

    ### Vis if selected ###
    if args.viz:
        all_coords_plot = {k:np.array(v) for k, v in all_coords.items()}
        split_mean = {}
        for split, split_coords in all_coords_plot.items():
            # shift to zero mean
            split_mean[split] = np.mean(split_coords, 0)
            all_coords_plot[split] = np.array(split_coords) - split_mean[split]

        fig = plt.figure(figsize=(6*len(splits)+2, 18))

        for i, split in enumerate(splits):
            ax = fig.add_subplot(2, math.ceil(len(splits)/2), i+1)
            ax.set_title(split)
            ax.set_xlabel('x [m]')
            ax.set_aspect('equal', 'box')
            if i == 0:
                ax.set_ylabel('y [m]')
            # img = plt.imread(img_dict[split]) # not currently working, needs alignment
            # ax.imshow(img)
            ax.scatter(all_coords_plot[split][:,0], all_coords_plot[split][:,1], 
                       c = all_colours[split], s = all_sizes[split])
            for poly in POLY_DICT[split]:
                xy = np.array(poly.exterior.xy) - split_mean[split].T.reshape(-1,1)
                ax.plot(*xy, 'k-')
            for poly in GROUND_EXCLUDE_DICT[split]:
                xy = np.array(poly.exterior.xy) - split_mean[split].T.reshape(-1,1)
                ax.plot(*xy, 'r-')
        
        plt.tight_layout()        
        plt.show()

    if args.query_requires_ground:
        ground_positives = "_ground-positives-required_"
    elif args.ground_aerial_positives_only:
        ground_positives = "_ground-aerial-only_"
    else:
        ground_positives = "_"
    train_file_baseline_basename = osp.join(save_dir, f"training_queries_CSWildPlaces_baseline{ground_positives}")
    test_file_base = osp.join(save_dir, "test_queries_CSWildPlaces_")
    construct_training_query_dict(df_train_baseline, train_file_baseline_basename)
    construct_training_query_dict(df_test, test_file_base, test_set=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, required = True, 
                        help='Root directory containing splits of CS-Wild-Places dataset')
    parser.add_argument('--save_dir', type = str, default = None, 
                        help='Directory to save training queries to, default is --root')
    parser.add_argument('--splits', nargs = '+', default = [], 
                        help='Splits (min 1) in root folder to process. Processes every folder in root if empty')
    parser.add_argument('--eval_thresh', type = float, default = 15,
                        help = 'Threshold of nearest database submap for choosing eval queries')
    parser.add_argument('--pos_thresh', type = float, required = True, 
                        help = 'Threshold (m) for positive training matches, default 0.5*radius')
    parser.add_argument('--neg_thresh', type = float, required = True, 
                        help = 'Threshold (m) for negative training matches, default 2*radius')
    parser.add_argument('--buffer_thresh', type = float, required = True, 
                        help = 'Threshold (m) from ground positives to keep as buffer zone, default 1*radius')
    parser.add_argument('--ignore_positives_poses', default = False, action = 'store_true', 
                        help = 'Prevents saving positive poses to pickle file (saves space if not pre-computing ICP)')
    parser.add_argument('--icp', default = False, action = 'store_true', 
                        help = 'Use ICP refinement for GT pose')
    parser.add_argument('--cache_submaps', default = False, action = 'store_true', 
                        help = 'Cache submaps to speed up ICP (disable if RAM limited)')
    parser.add_argument('--query_requires_ground', default = False, action = 'store_true', 
                        help = 'Only save training queries that either are from the ground, or have at least 1 ground positive (to dissuade massive aerial bias)')
    parser.add_argument('--ground_aerial_positives_only', default = False, action = 'store_true', 
                        help = 'Only save training queries and positives that contain ground/aerial matches. Removes ground/ground or aerial/aerial positives from training.')
    parser.add_argument('--viz', default = False, action = 'store_true', 
                        help = 'Enable visualisations of train/test splits')
    args = parser.parse_args()

    if args.query_requires_ground:
        raise NotImplementedError("Currently disabled due to poor implementation, and is now handled in the batch sampler")

    if args.query_requires_ground and args.ground_aerial_positives_only:
        print("[WARNING] --ground_aerial_positives_only will supersede --query_requires_ground, thus the latter will have no effect")
        args.query_requires_ground = False

    assert not (args.ignore_positives_poses and args.icp), (
        "ICP not required if not saving positives poses to pickle"
    )
    
    args.save_dir = args.root if args.save_dir is None else args.save_dir
    
    print(args)
    main()
