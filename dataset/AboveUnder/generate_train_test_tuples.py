"""
Generate train, test, and eval tuples for the Above-Under dataset. Generates 
baseline and refined set of training queries, for testing out-of-domain 
performance. 

By Ethan Griffiths (Data61, Pullenvale)
"""
from os import path, listdir, makedirs
import pickle
import random
import math
import argparse
import warnings

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn', disables annoying warning
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.base_datasets import TrainingTuple

CLOUD_DIR = 'clouds/'
POSES_FILE = 'poses.csv'

# Initialise random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
MARKER_SIZES = {'aerial':4, 'ground':8}
# img_dict = {'QCAT':'maps/qcat.png', 'Samford':'maps/samford.png', 'Robson':'maps/robson.png'}

VAL_SPLIT = 'Karawatha'    # split to use for validation during training
BASELINE_SPLITS = ['Karawatha', 'Venman']  # splits in baseline train set

### POLYGONS (easting, northing)
# For QCAT
p1 = Polygon([(491013, 6955331), (491013, 6955353), (491090, 6955353), (491090, 6955294)])
# For Samford
p2 = Polygon([(487542, 6970331), (487542, 6970464), (487648, 6970464), (487648, 6970331)])
p3 = Polygon([(487831, 6970162), (487831, 6970230), (487982, 6970230), (487982, 6970162)])
# For Robson
p4 = Polygon([(354200,8106640), (354200,8106870), (354210,8106870), (354210,8106640)])
# For Beetaloo
p5 = Polygon([(365747,8195521), (365747,8195625), (365884,8195625), (365884,8195521)])
# For Karawatha (same as wild places, but transformed to UTM frame with relative_transform.txt)
p6 = Polygon([(5.07019178e+05,  6.94265947e+06), (5.07469178e+05,  6.94266009e+06),
              (5.07469479e+05,  6.94244209e+06), (5.07019479e+05,  6.94244147e+06)])
p7 = Polygon([(5.06953336e+05,  6.94326937e+06), (5.07094336e+05,  6.94326957e+06),
              (5.07094605e+05,  6.94307457e+06), (5.06953605e+05,  6.94307438e+06)])
p8 = Polygon([(5.06655775e+05,  6.94295096e+06), (5.06656139e+05,  6.94268796e+06),
              (5.06848138e+05,  6.94268823e+06), (5.06847775e+05,  6.94295123e+06)])
# p6 = Polygon([(-150, 8), (300,8), (300,-210), (-150,-210)])
# p7 = Polygon([(-215,618), (-74,618), (-74,423), (-215,423)])
# p8 = Polygon([(-513,300), (-513,37), (-321,37), (-321,300)])

POLY_DICT = {'QCAT':[p1], 'Samford':[p2, p3], 'Robson':[p4], 'Beetaloo':[p5], 
             'Karawatha':[p6,p7,p8]}
###

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

def check_in_test_set(easting, northing, test_polygons, run_type, 
                      test_queries : KDTree = None):
    """
    Return which split submap is in, given polygon containing test queries. Run
    twice, first to get the ground test queries, and second to find all submaps
    in the buffer zone.
    """
    split = 'train'
    point = Point(easting, northing)
    for poly in test_polygons:
        if poly.contains(point) and run_type == 'ground':
            split = 'test'
            return split
    if test_queries is not None:
        coord = np.array([easting, northing]).reshape(1, -1)
        num_matches = test_queries.query_radius(coord, args.buffer_thresh, count_only=True)
        if num_matches[0] > 0:
            split = 'buffer'
    return split

def construct_training_query_dict(df_centroids, filename_base, test_set=False, v2_only=False):
    """
    Create training query dictionaries with v1 and v2 (MinkLoc3Dv2) formats.
    """
    run_str = 'test' if test_set else 'training'
    print(f"Computing {run_str} queries...")
    file_v1 = filename_base + 'v1.pickle'
    file_v2 = filename_base + 'v2.pickle'
    tree = KDTree(df_centroids[['easting','northing']])
    ind_nn = tree.query_radius(
        df_centroids[['easting','northing']], r=args.pos_thresh
    )
    # store ground indices to remove from test set positives (ONLY EVAL WITH GROUND QUERY, AERIAL DATABASE)
    ind_ground = np.array([i for i, x in enumerate(df_centroids['file'].to_numpy()) if 'ground' in x])
    ind_r = tree.query_radius(
        df_centroids[['easting','northing']], r=args.neg_thresh
    )
    ind_df_centroids = df_centroids.index.values.tolist()
    queries_v1 = {}
    queries_v2 = {}
    num_queries_skipped = { # hardcoded for now but it's just for debugging
        'Beetaloo':0, 'Karawatha':0, 'QCAT':0, 'Robson':0, 'Samford':0
    }
    for anchor_ndx in tqdm(range(len(ind_nn))):
        anchor_pos = np.array(
            df_centroids.iloc[anchor_ndx][['easting', 'northing']], dtype=np.float64
        )
        query = df_centroids.iloc[anchor_ndx]['file']
        split = str.split(query, '/')[0]    # first component is split
        # Extract timestamp from the filename
        scan_filename = path.split(query)[1]
        timestamp = str(path.splitext(scan_filename)[0])
        
        positives = np.setdiff1d(ind_nn[anchor_ndx], [anchor_ndx])
        # remove queries with no ground positives, or remove all aerial queries if creating test set
        if (test_set and 'aerial' in query) or (
            args.query_requires_ground 
            and 'aerial' in query
            and not any(
                ['ground' in file for file in df_centroids.iloc[positives]['file']]
            )
        ):
            num_queries_skipped[split] += 1
            positives = np.array([])
            negatives = np.array([])
            non_negatives = np.array([])
        else:
            negatives = np.setdiff1d(ind_df_centroids, ind_r[anchor_ndx])
            non_negatives = np.sort(ind_r[anchor_ndx])            
            # remove ground positives/negatives from test set
            if test_set and 'ground' in query:
                positives = np.setdiff1d(positives, ind_ground)
                negatives = np.setdiff1d(negatives, ind_ground)
                non_negatives = np.union1d(non_negatives, ind_ground)
            np.random.shuffle(negatives)

        if not v2_only:
            queries_v1[anchor_ndx] = {
                "query":query, "positives":positives.tolist(), 
                "negatives":negatives.tolist()
            }
        queries_v2[anchor_ndx] = TrainingTuple(
            id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
            positives=positives, non_negatives=non_negatives, 
            position=anchor_pos
        )
    
    print(f"Queries skipped per split:")
    num_queries_skipped_total = 0
    for split, num in num_queries_skipped.items():
        print(f"{split}: {num}")
        num_queries_skipped_total += num
    print(f"Final number of {run_str} queries: {len(queries_v2) - num_queries_skipped_total}/{len(queries_v2)}")
    if not v2_only:
        output_to_file(queries_v1, file_v1)
    output_to_file(queries_v2, file_v2)
    
    return True

def construct_query_and_database_sets(database_trees, database_sets, test_sets, filename_base):
    print("Saving queries and database...")
    radius = args.radius_max    
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
                    index = tree.query_radius(coor, r=radius)
                    # indices of the positive matches in database i of each query (key) in test set j
                    test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, file_db)
    output_to_file(test_sets, file_query)
    
    return True

def format_df(df_locations: pd.DataFrame, clouds_relpath: str):
    # Fix column names for above-under
    df_locations.rename(columns={'x':'easting', 'y':'northing'}, inplace=True)
    df_locations = df_locations[['timestamp','easting','northing']]
    
    # Create filepath from timestamp
    df_locations.loc[:,'timestamp'] = clouds_relpath \
                                      + df_locations.loc[:,'timestamp'] \
                                      + '.pcd'
    df_locations.rename(columns={'timestamp':'file'}, inplace=True)
    
    return df_locations

def main():
    root_dir = args.root
    save_dir = args.save_dir
    splits = args.splits
    v2_only = args.v2_only
    
    if not path.exists(save_dir):
        makedirs(save_dir)

    # Initialize pandas DataFrame
    df_train_baseline = pd.DataFrame(columns=['file','easting','northing'])
    df_train_refined = pd.DataFrame(columns=['file','easting','northing'])
    df_test = pd.DataFrame(columns=['file','easting','northing'])
    test_queries = []

    # For visualisations
    all_coords = {}
    all_colours = {}
    all_sizes = {}

    # Find splits in dataset folder
    if len(splits) == 0:
        splits = sorted(listdir(root_dir))
        splits = [x for x in splits if path.isdir(path.join(root_dir, x))]
    for split in splits:
        print(f'Processing {split}')
        if split not in POLY_DICT.keys():
            warnings.warn('Split is not recognised, no test areas are '
                          'associated with it. Ignoring...')
            continue
        all_coords[split] = []
        all_colours[split] = []
        all_sizes[split] = []
        folders = sorted(listdir(path.join(root_dir, split)))
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
                path.join(root_dir, split, folder, POSES_FILE), 
                sep=',', 
                dtype={'timestamp':str}
            )
            
            # Get easting and northing
            coords = df_locations[['x','y']].to_numpy()

            # Find ground queries
            for row in coords:
                row_split = check_in_test_set(row[0], row[1], POLY_DICT[split], 
                                            run_type, None)
                if row_split == 'test':
                    test_queries.append(row)                
        
        test_queries_tree = KDTree(test_queries)
        print('Done')
        
        # Reset counters
        test_counter = dict.fromkeys(['aerial','ground'], 0)
        buffer_counter = dict.fromkeys(['aerial','ground'], 0)
        train_counter = dict.fromkeys(['aerial','ground'], 0)

        database_trees = []
        database_sets = []
        test_sets = []

        # Gather submaps from each folder in split        
        print(f'Processing submaps... ', end='')
        for folder in folders:
            df_database = pd.DataFrame(columns=['file','easting','northing'])
            database_dict = {}
            test_dict = {}
            if 'aerial' in folder:
                run_type = 'aerial' 
            elif 'ground' in folder:
                run_type = 'ground'
            else:
                raise AssertionError(f'Invalid folder "{folder}", '
                                    'must contain aerial or ground in name')
            
            df_locations = pd.read_csv(
                path.join(root_dir, split, folder, POSES_FILE),
                sep=',',
                dtype={'timestamp':str}
            )
            
            # Fix column names and filenames
            clouds_relpath = path.join(split, folder, CLOUD_DIR)
            df_locations = format_df(df_locations, clouds_relpath)
            
            # Sort submaps by train, test, and buffer set
            for _, row in tqdm(df_locations.iterrows(), desc=folder, 
                                total=len(df_locations)):
                assert path.isfile(path.join(root_dir, row['file'])), \
                    f"No associated submap for pose: {row['file']}"
                all_coords[split].append(row[['easting','northing']])
                all_sizes[split].append(MARKER_SIZES[run_type])
                row_split = check_in_test_set(row['easting'], row['northing'], 
                                            POLY_DICT[split], run_type, 
                                            test_queries_tree)
                if row_split == 'test':
                    if split == VAL_SPLIT:  # test queries only consider Karawatha, for consistency with other models (as minkloc3dv2 is the only to validate using the test query tuple)
                        df_test.loc[len(df_test)] = row
                    test_dict[len(test_dict.keys())] = {
                        'query':row['file'],
                        'easting':row['easting'],
                        'northing':row['northing']
                    }
                    test_counter[run_type] += 1
                    all_colours[split].append([1,0,0])
                elif row_split == 'buffer':
                    buffer_counter[run_type] += 1
                    all_colours[split].append([1,165/255,0])
                else:
                    if split in BASELINE_SPLITS:
                        df_train_baseline.loc[len(df_train_baseline)] = row
                    df_train_refined.loc[len(df_train_refined)] = row
                    train_counter[run_type] += 1
                    all_colours[split].append([0,0,1])
                    
                if run_type == 'aerial':    # all aerial submaps form database
                    if split == VAL_SPLIT:
                        df_test.loc[len(df_test)] = row
                    df_database.loc[len(df_database)] = row
                    database_dict[len(database_dict.keys())] = {
                        'query':row['file'],
                        'easting':row['easting'],
                        'northing':row['northing']
                    }
            database_tree = KDTree(df_database[['easting','northing']]) if not df_database.empty else None
            database_trees.append(database_tree)
            database_sets.append(database_dict)
            test_sets.append(test_dict)

        print('Done')
        # save query/db pickles
        filename_base = path.join(save_dir, f"above-under_{split}_evaluation")
        construct_query_and_database_sets(database_trees, database_sets, test_sets, filename_base)        
        
        print(f'{split} stats:\n'
            f'\tTraining submaps - {train_counter["aerial"] + train_counter["ground"]} '
            f'({train_counter["aerial"]} aerial, {train_counter["ground"]} ground)\n'
            f'\tTest submaps     - {test_counter["aerial"] + test_counter["ground"]} '
            f'({test_counter["aerial"]} aerial, {test_counter["ground"]} ground)\n'
            f'\tBuffer submaps   - {buffer_counter["aerial"] + buffer_counter["ground"]} '
            f'({buffer_counter["aerial"]} aerial, {buffer_counter["ground"]} ground)')

    print(f"\nTotal number of potential baseline training submaps: "
          f"{len(df_train_baseline['file'])}")
    print(f"Total number of potential refined training submaps: "
          f"{len(df_train_refined['file'])}")
    print(f"Total number of test query submaps: {len(test_queries)}")

    ### Vis if selected ###
    if args.viz == True:
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
                xy = np.array(poly.exterior.xy) \
                     - split_mean[split].T.reshape(-1,1)
                ax.plot(*xy, 'k-')
        
        plt.tight_layout()        
        plt.show()

    # ground_positives = "_ground-positives-req_" if args.query_requires_ground else "_ground-positives-not-req_"
    # train_file_baseline_base = path.join(save_dir, f"training_queries_above-under_baseline{ground_positives}")
    # train_file_refined_base = path.join(save_dir, f"training_queries_above-under_refined{ground_positives}")
    train_file_baseline_basename = path.join(save_dir, f"training_queries_above-under_baseline_")
    train_file_refined_basename = path.join(save_dir, f"training_queries_above-under_refined_")
    test_file_base = path.join(save_dir, "test_queries_above-under_")
    construct_training_query_dict(df_train_baseline, train_file_baseline_basename, v2_only=v2_only)
    construct_training_query_dict(df_train_refined, train_file_refined_basename, v2_only=v2_only)
    construct_training_query_dict(df_test, test_file_base, test_set=True, v2_only=v2_only)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, required = True, 
                        help='Root directory containing splits of above_under dataset (ideally after postprocessing)')
    parser.add_argument('--save_dir', type = str, default = None, 
                        help='Directory to save training queries to, default is --root')
    parser.add_argument('--splits', nargs = '+', default = [], 
                        help='Splits (min 1) in root folder to process. Processes every folder in root if empty')
    parser.add_argument('--radius_max', type = float, default = 30, 
                        help = 'Max radius (m) of submaps')
    parser.add_argument('--pos_thresh', type = float, default = -1, 
                        help = 'Threshold (m) for positive matches, default 0.5*radius')
    parser.add_argument('--neg_thresh', type = float, default = -1, 
                        help = 'Threshold (m) for negative matches, default 2*radius')
    parser.add_argument('--buffer_thresh', type = float, default = -1, 
                        help = 'Threshold (m) from ground positives to keep as buffer zone, default 2*radius')
    parser.add_argument('--query_requires_ground', default = False, action = 'store_true', 
                        help = 'Only save training queries that either are from the ground, or have at least 1 ground positive (to dissuade massive aerial bias)')
    parser.add_argument('--v2_only', default = False, action = 'store_true', 
                        help = 'Only save queries in v2 format (Minkloc3D style)')
    parser.add_argument('--viz', default = False, action = 'store_true', 
                        help = 'Enable visualisations of train/test splits')
    args = parser.parse_args()
        
    # args.query_requires_ground = True   # Forcing all queries to require ground positives for now
    
    if args.pos_thresh < 0:
        args.pos_thresh = 0.5 * args.radius_max
    if args.neg_thresh < 0:
        args.neg_thresh = 2 * args.radius_max
    if args.buffer_thresh < 0:
        args.buffer_thresh = 2 * args.radius_max
    
    args.save_dir = args.root if args.save_dir is None else args.save_dir
    
    print(args)
    main()
