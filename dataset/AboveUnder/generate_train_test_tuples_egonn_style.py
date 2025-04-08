"""
Generate train, test, and eval tuples for the QCAT datasets. 

By Ethan Griffiths (Data61, Pullenvale)
"""
import argparse
import math
import pickle
import random
import sys
import time
import warnings
from os import listdir, makedirs, path
from typing import Dict, List

import numpy as np
import open3d as o3d
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn', disables annoying warning

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point, Polygon
from sklearn.neighbors import KDTree
from tqdm import tqdm

sys.path.append(path.join(path.dirname(__file__), '../..'))
from dataset.AboveUnder.AboveUnder_raw import AboveUnderPointCloudLoader

CLOUD_DIR = 'clouds/'
POSES_FILE = "poses.csv"

# Initialise random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
MARKER_SIZES = {'aerial':4, 'ground':8}
DF_COLUMNS = ['file','easting','northing', 'x','y','z','qx','qy','qz','qw']

# VAL_SPLITS = ['Karawatha']   # splits to use for validation during training
# BASELINE_SPLITS = ['Karawatha']  # splits in baseline train set
VAL_SPLITS = ['Karawatha', 'Venman']   # splits to use for validation during training
BASELINE_SPLITS = ['Karawatha', 'Venman']  # splits in baseline train set

### POLYGONS (easting, northing)
## BELOW ARE THE OLD TEST REGIONS FOR QCAT, SAMFORD, ROBSON, BUT NOW THESE ENTIRE
## SETS ARE BEING USED DURING TEST TIME
# For QCAT
# p1 = Polygon([(491013, 6955331), (491013, 6955353), (491090, 6955353), (491090, 6955294)])
# # For Samford
# p2 = Polygon([(487542, 6970331), (487542, 6970464), (487648, 6970464), (487648, 6970331)])
# p3 = Polygon([(487831, 6970162), (487831, 6970230), (487982, 6970230), (487982, 6970162)])
# For Robson
# p4 = Polygon([(354200,8106640), (354200,8106870), (354210,8106870), (354210,8106640)])
# For Beetaloo
# p5 = Polygon([(365747,8195521), (365747,8195625), (365884,8195625), (365884,8195521)])

# For Karawatha (same as wild places, but transformed to UTM frame with relative_transform.txt)
# p6 = Polygon([(-150, 8), (300,8), (300,-210), (-150,-210)])  # original frame
# p7 = Polygon([(-215,618), (-74,618), (-74,423), (-215,423)])
# p8 = Polygon([(-513,300), (-513,37), (-321,37), (-321,300)])
# p6 = Polygon([(5.07019178e+05,  6.94265947e+06), (5.07469178e+05,  6.94266009e+06),  # OLD UTM TRANSFORM
#               (5.07469479e+05,  6.94244209e+06), (5.07019479e+05,  6.94244147e+06)])
# p7 = Polygon([(5.06953336e+05,  6.94326937e+06), (5.07094336e+05,  6.94326957e+06),
#               (5.07094605e+05,  6.94307457e+06), (5.06953605e+05,  6.94307438e+06)])
# p8 = Polygon([(5.06655775e+05,  6.94295096e+06), (5.06656139e+05,  6.94268796e+06),
#               (5.06848138e+05,  6.94268823e+06), (5.06847775e+05,  6.94295123e+06)])

## NEW TEST REGIONS WHICH COVER THE ENTIRE SPLITS
# For QCAT
p1 = Polygon([(490500, 6955000), (490500, 6956000), (491500, 6956000), (491500, 6955000)])
# For Samford
p2 = Polygon([(487000, 6969000), (487000, 6971000), (489000, 6971000), (489000, 6969000)])
# For Robson
p4 = Polygon([(353000,8106000), (353000,8108000), (355000,8108000), (355000,8106000)])
# For Beetaloo
p5 = Polygon([(365000,8195000), (365000,8197000), (367000,8197000), (367000,8195000)])

# NEW UTM TRANSFORM
# Karawatha
p6 = Polygon([(507018.60467,6942659.3756), (507468.60473,6942659.6724),
              (507468.74853,6942441.6724), (507018.74850,6942441.3756)])
p7 = Polygon([(506953.20227,6943269.3327), (507094.20227,6943269.4257),
              (507094.33093,6943074.4257), (506953.33090,6943074.3327)])
p8 = Polygon([(506655.41198,6942951.1361), (506655.58551,6942688.1361),
              (506847.58554,6942688.2628), (506847.41204,6942951.2627)])
# Venman
p9 = Polygon([(519331.85162354,6943652.20440674), (519331.19000244,6943778.20266724),
              (519485.18786621,6943779.01129150), (519494.35580444,6943747.05899048),
              (519607.18621826,6943779.65188599), (519607.84783936,6943653.65362549)])
p10 = Polygon([(519722.31359863,6943565.25347900), (519722.54461670,6943521.25408936),
               (519495.54779053,6943520.06213379), (519495.31674194,6943564.06152344)])
p11 = Polygon([(519737.04788208,6943806.33413696), (519894.04573059,6943807.15850830),
               (519941.41265869,6943737.40628052), (519940.15832520,6943595.39773560),
               (519738.16110229,6943594.33709717)])

# POLY_DICT = {'QCAT':[p1], 'Samford':[p2,p3], 'Robson':[p4], 'Beetaloo':[p5], 
#              'Karawatha':[p6,p7,p8]}  # OLD SPLITS
POLY_DICT = {'QCAT':[p1], 'Samford':[p2], 'Robson':[p4], 'Beetaloo':[p5], 
             'Karawatha':[p6,p7,p8], 'Venman':[p9, p10, p11]}  # NEW SPLITS
###

# # Exclude Karawatha region with high drift
# kara_exclude1 = Polygon([[507460, 6942602], [507930, 6942602], [507930, 6941940], 
#                          [506990, 6941940], [506990, 6942390], [507460, 6942390]])
# EXCLUDE_DICT = {'QCAT':[], 'Samford':[], 'Robson':[], 'Beetaloo':[], 
#                 'Karawatha':[kara_exclude1], 'QCAT_reloc':[], 'EPE_Hill_reloc':[],
#                 'QCAT_reloc_v2':[]}
EXCLUDE_DICT = {'QCAT':[], 'Samford':[], 'Robson':[], 'Beetaloo':[], 
                'Karawatha':[], 'Venman':[]}
###


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, pose: np, positives_poses: Dict[int, np.ndarray] = None):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # pose: pose as 4x4 matrix
        # positives_poses: relative poses of positive examples refined using ICP
        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.pose = pose
        self.positives_poses = positives_poses


class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: float, rel_scan_filepath: str, position: np.array, pose: np.array = None):
        # position: x, y position in meters
        # pose: 6 DoF pose (as 4x4 pose matrix)
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position
        self.pose = pose

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position, self.pose


class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions


def icp(anchor_pc, positive_pc, transform: np.ndarray = None, point2plane: bool = False,
        inlier_dist_threshold: float = 1.2, max_iteration: int = 200):
    # transform: initial alignment transform
    if transform is not None:
        transform = transform.astype(float)

    voxel_size = 0.1
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(anchor_pc)
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(positive_pc)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    if point2plane:
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    if transform is not None:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold, transform,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

def quaternion_to_rot(quat_transform) -> np.ndarray:
    """
    x, y, z, qx, qy, qz, qw -> SE3 pose
    """
    # x, y, z, qx, qy, qz, qw format
    xyz = quat_transform[:3]
    quaternion = quat_transform[3:]

    r = R.from_quat(quaternion)
    rot_matrix = r.as_matrix()
    trans = np.concatenate([rot_matrix, xyz.reshape(3,1)], axis = 1)
    trans = np.concatenate([trans, np.array([[0.,0.,0.,1.]])], axis = 0)
    return trans

def rot_to_quaternion(rot_transform):
    """
    SE3 pose -> x, y, z, qx, qy, qz, qw
    """
    # qx, qy, qz, qw format
    r = R.from_matrix(rot_transform[:3,:3])
    quat = r.as_quat()
    xyz = rot_transform[:3,3]
    return xyz, quat

def relative_pose(m1, m2):
    # SE(3) pose is 4x 4 matrix, such that
    # Pw = [R | T] @ [P]
    #      [0 | 1]   [1]
    # where Pw are coordinates in the world reference frame and P are coordinates in the camera frame
    # m1: coords in camera/lidar1 reference frame -> world coordinate frame
    # m2: coords in camera/lidar2 coords -> world coordinate frame
    # returns: relative pose of the first camera with respect to the second camera
    #          transformation matrix to convert coords in camera/lidar1 reference frame to coords in
    #          camera/lidar2 reference frame
    #
    m = np.linalg.inv(m2) @ m1
    # # !!!!!!!!!! Fix for relative pose !!!!!!!!!!!!!
    # m[:3, 3] = -m[:3, 3]
    return m

def get_timestamp_from_file(file):
    timestamp = str(path.splitext(path.split(file)[-1])[0])
    return timestamp

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

def check_in_test_set(easting, northing, split, run_type, 
                      test_queries_tree : KDTree = None):
    """
    Return which set submap is in, given polygon containing test queries. Run
    twice, first to get the ground test queries, and second to find all submaps
    in the buffer zone.
    """
    assert run_type in ('ground','aerial'), "run_type must be in ('ground','aerial')"
    submap_set = 'train'
    point = Point(easting, northing)
    test_polygons = POLY_DICT[split]
    exclude_polygons = EXCLUDE_DICT[split]
    for poly in test_polygons:
        if poly.contains(point):
            if run_type == 'ground':
                submap_set = 'test'
            elif run_type == 'aerial':
                submap_set = 'buffer'
            return submap_set
    if test_queries_tree is not None:
        coord = np.array([easting, northing]).reshape(1, -1)
        num_matches = test_queries_tree.query_radius(coord, args.buffer_thresh, count_only=True)
        if num_matches[0] > 0:
            submap_set = 'buffer'
            
    # check in exclusion set
    if len(exclude_polygons) > 0:
        for poly in exclude_polygons:
            if poly.contains(point):
                submap_set = 'buffer'
        
    return submap_set

def cache_pcs(df_centroids: pd.DataFrame) -> dict:
    """
    Pre-load all submaps in the train/test set into a dictionary, to speed up
    ICP computation. Dictionary contains numpy submaps idxed with same idx as
    df_centroids.
    """
    pc_dict = {}
    pc_loader = AboveUnderPointCloudLoader()
    for ndx, row in tqdm(df_centroids.iterrows(), total=len(df_centroids),
                         desc='Caching submaps for ICP'):
        pc = pc_loader.read_pc(path.join(args.root, row["file"]))
        pc_dict[ndx] = pc
    
    return pc_dict

def construct_training_query_dict(df_centroids, filename_base, test_set=False):
    """
    Create training query dictionaries with EgoNN format.
    """
    run_str = 'test' if test_set else 'training'
    print(f"Computing {run_str} queries...")
    pc_loader = AboveUnderPointCloudLoader()
    file_v2 = filename_base + 'v2.pickle'
    tree = KDTree(df_centroids[['easting','northing']])
    ind_nn = tree.query_radius(
        df_centroids[['easting','northing']], r=args.pos_thresh
    )
    # store ground and aerial indices to remove from test set positives (ONLY EVAL WITH GROUND QUERY, AERIAL DATABASE)
    cloud_files = df_centroids['file'].to_numpy()
    ind_ground = np.array([i for i, x in enumerate(cloud_files) if 'ground' in x])
    ind_aerial = np.array([i for i, x in enumerate(cloud_files) if 'aerial' in x])
    ind_non_neg = tree.query_radius(
        df_centroids[['easting','northing']], r=args.neg_thresh
    )
    ind_df_centroids = df_centroids.index.values.tolist()
    queries_v2 = {}
    num_queries_skipped = {
        split:0 for split in POLY_DICT.keys()
    }
    # Cache submaps for ICP computation (saves IO in the long run)
    if args.icp and args.cache_submaps:
        pc_dict = cache_pcs(df_centroids)    

    count_no_positives = 0
    fitness_l = []
    inlier_rmse_l = []
    for anchor_ndx in tqdm(range(len(ind_nn)), desc='Computing'):
        anchor_pose = quaternion_to_rot(np.array(
            df_centroids.iloc[anchor_ndx][['x','y','z','qx','qy','qz','qw']], dtype=np.float64
        ))
        query = df_centroids.iloc[anchor_ndx]['file']
        split = str.split(query, '/')[0]    # first component is split
        # Extract timestamp from the filename
        timestamp = get_timestamp_from_file(query)
        
        positives = np.setdiff1d(ind_nn[anchor_ndx], [anchor_ndx])   
        non_negatives = np.sort(ind_non_neg[anchor_ndx])
        
        # ICP pose refinement
        positive_poses = {}
                
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
            non_negatives = np.array([])
        # remove ground positives/negatives from test set
        elif test_set and 'ground' in query: 
            positives = np.setdiff1d(positives, ind_ground)
            non_negatives = np.union1d(non_negatives, ind_ground)

        # remove ground/ground and aerial/aerial positives
        if args.ground_aerial_positives_only:
            if 'ground' in query:
                positives = np.setdiff1d(positives, ind_ground)
                non_negatives = np.union1d(non_negatives, ind_ground)
            elif 'aerial' in query:
                positives = np.setdiff1d(positives, ind_aerial)
                non_negatives = np.union1d(non_negatives, ind_aerial)                
        
        if len(positives) == 0:
            count_no_positives += 1
        
        if args.icp:
            if args.cache_submaps:
                anchor_pc = pc_dict[anchor_ndx]
            else:
                anchor_pc = pc_loader.read_pc(path.join(args.root, df_centroids.iloc[anchor_ndx]["file"]))

        # tic = time.time()
        for positive_ndx in positives:
            positive_pose = quaternion_to_rot(np.array(
                df_centroids.iloc[positive_ndx][['x','y','z','qx','qy','qz','qw']], dtype=np.float64
            ))
            # Compute initial relative pose
            transform = relative_pose(anchor_pose, positive_pose)
            positive_poses[positive_ndx] = transform
            # print(f'Dist between anchor and positve {positive_ndx} is: ', np.sqrt(transform[0][3]*transform[0][3] + transform[1][3]*transform[1][3]))
            if args.icp:
                # Refine the pose using ICP
                if args.cache_submaps:
                    positive_pc = pc_dict[positive_ndx]
                else:
                    positive_pc = pc_loader.read_pc(path.join(args.root, df_centroids.iloc[positive_ndx]["file"]))
                m, fitness, inlier_rmse = icp(anchor_pc, positive_pc, transform)
                fitness_l.append(fitness)
                inlier_rmse_l.append(inlier_rmse)
                positive_poses[positive_ndx] = m
        # icp_time = time.time() - tic
        # print(f"{len(positives)} positives took {icp_time:.1f} secs", flush=True) 

        queries_v2[anchor_ndx] = TrainingTuple(
            id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
            positives=positives, non_negatives=non_negatives, pose=anchor_pose,
            positives_poses = positive_poses
        )
    
    print(f"Queries with no positives: {count_no_positives}")
    print("Queries skipped per split:")
    num_queries_skipped_total = 0
    for split, num in num_queries_skipped.items():
        print(f"\t{split}: {num}")
        num_queries_skipped_total += num
    if args.icp:
        avg_fitness = np.mean(fitness_l)
        avg_inlier_rmse = np.mean(inlier_rmse_l)
        print("ICP Results:")
        print(f"\tAvg fitness: {avg_fitness}")
        print(f"\tAvg inlier RMSE: {avg_inlier_rmse}")
    print(f"Final number of {run_str} queries: {len(queries_v2) - count_no_positives}/{len(queries_v2)}")
    if not args.debug:
        output_to_file(queries_v2, file_v2)        
    
    return True

def construct_eval_sets(database_set, test_set, filename_eval) -> int:
    print("Saving eval set...")
    threshold = args.eval_thresh
    map_pos = np.zeros((len(database_set), 2), dtype=np.float32)
    for ndx, e in enumerate(database_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    # Filters out query elements without a corresponding map element within eval_thresh threshold
    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(test_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    num_filtered_queries = len(filtered_query_set)
    print(f"{count_ignored}/{len(test_set)} query elements ignored - not having corresponding map element within {threshold} [m] radius")

    eval_set = EvaluationSet(filtered_query_set, database_set)
    if not args.debug:
        eval_set.save(filename_eval)
    
    return num_filtered_queries

def format_df(df_locations: pd.DataFrame, clouds_relpath: str):
    # Fix column names for above-under
    df_locations['easting'] = df_locations.loc[:, 'x']
    df_locations['northing'] = df_locations.loc[:, 'y']
    # Leave x and y columns as we need to save pose of each submap
    # df_locations.rename(columns={'x':'easting', 'y':'northing'}, inplace=True)
    df_locations = df_locations[['timestamp', *DF_COLUMNS[1:]]]
    
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
    
    if not path.exists(save_dir):
        makedirs(save_dir)

    # Initialize pandas DataFrame
    df_train_baseline = pd.DataFrame(columns=DF_COLUMNS)
    df_train_refined = pd.DataFrame(columns=DF_COLUMNS)
    df_test = pd.DataFrame(columns=DF_COLUMNS)
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
            for run in args.exclude_runs:
                if run in folder:
                    print(f"Skipping {folder}")
                    folders.remove(folder)
        
        # Separate ground folders
        ground_folders = [folder for folder in folders if 'ground' in folder]
        
        # Determine ground test queries
        print('Getting ground queries... ', end='')
        for folder in ground_folders:
            run_type = 'ground'
            df_locations = pd.read_csv( # ts,x,y,z,qx,qy,qz,qw
                path.join(root_dir, split, folder, POSES_FILE),
                sep=',', 
                dtype={'timestamp':str}
            )
            
            # Get easting and northing
            coords = df_locations[['x','y']].to_numpy()

            # Find ground queries
            for row in coords:
                row_split = check_in_test_set(row[0], row[1], split, 
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

        # database_trees = []
        database_set = []
        test_set = []

        # Gather submaps from each folder in split        
        print(f'Processing submaps... ', end='')
        for folder in folders:
            df_database = pd.DataFrame(columns=DF_COLUMNS)
            database_tuples = []
            test_tuples = []
            if 'aerial' in folder:
                run_type = 'aerial' 
            elif 'ground' in folder:
                run_type = 'ground'
            else:
                raise AssertionError(f'Invalid folder "{folder}", '
                                    'must contain aerial or ground in name')
            
            df_locations = pd.read_csv( # ts,x,y,z,qx,qy,qz,qw
                path.join(root_dir, split, folder, POSES_FILE),
                sep=',',
                dtype={'timestamp':str}
            )
            
            # Fix column names and filenames
            clouds_relpath = path.join(split, folder, CLOUD_DIR)
            df_locations = format_df(df_locations, clouds_relpath)
            
            if args.debug:
                df_locations = df_locations.iloc[::100].reset_index(drop=True)
            
            # Sort submaps by train, test, and buffer set
            for _, row in tqdm(df_locations.iterrows(), desc=folder, 
                                total=len(df_locations)):
                assert path.isfile(path.join(root_dir, row['file'])), \
                    f"No associated submap for pose: {row['file']}"
                all_coords[split].append(row[['easting','northing']])
                all_sizes[split].append(MARKER_SIZES[run_type])
                row_split = check_in_test_set(row['easting'], row['northing'], 
                                              split, run_type, 
                                              test_queries_tree)
                if row_split == 'test':
                    if split in VAL_SPLITS:  # test queries only consider one split, for consistency with other models (as minkloc3dv2 is the only to validate using the test query tuple)
                        df_test.loc[len(df_test)] = row
                    test_tuple = EvaluationTuple(
                        timestamp=get_timestamp_from_file(row['file']),
                        rel_scan_filepath=row['file'],
                        pose=quaternion_to_rot(np.array(row[['x','y','z','qx','qy','qz','qw']], dtype=np.float64)),
                        position=np.array(row[['easting','northing']], dtype=np.float64)
                    )
                    test_tuples.append(test_tuple)
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
                    if split in VAL_SPLITS:
                        df_test.loc[len(df_test)] = row
                    df_database.loc[len(df_database)] = row
                    database_tuple = EvaluationTuple(
                        timestamp=get_timestamp_from_file(row['file']),
                        rel_scan_filepath=row['file'],
                        pose=quaternion_to_rot(np.array(row[['x','y','z','qx','qy','qz','qw']], dtype=np.float64)),
                        position=np.array(row[['easting','northing']], dtype=np.float64)
                    )
                    database_tuples.append(database_tuple)
            # database_tree = KDTree(df_database[['easting','northing']]) if not df_database.empty else None
            # database_trees.append(database_tree)
            database_set.extend(database_tuples)
            test_set.extend(test_tuples)

        print(f'Done {split}')
        # save query/db pickles
        num_filtered_queries = 0
        if len(test_set) > 0:
            filename_eval = path.join(save_dir, f"above-under_{split}_evaluation.pickle")
            num_filtered_queries = construct_eval_sets(database_set, test_set, filename_eval)
        
        print(
            f'{split} stats:\n'
            f'  Training submaps     - {train_counter["aerial"] + train_counter["ground"]} '
            f'({train_counter["aerial"]} aerial, {train_counter["ground"]} ground)\n'
            f'  Test submaps         - {test_counter["aerial"] + test_counter["ground"]} '
            f'({test_counter["aerial"]} aerial, {test_counter["ground"]} ground)\n'
            f'  Buffer submaps       - {buffer_counter["aerial"] + buffer_counter["ground"]} '
            f'({buffer_counter["aerial"]} aerial, {buffer_counter["ground"]} ground)\n'
            f'  Eval ground queries  - {num_filtered_queries} (from {len(test_set)} possible)\n'
            f'  Eval aerial database - {len(database_set)}'
        )

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
            ax = fig.add_subplot(min(len(splits),2), math.ceil(len(splits)/2), i+1)
            ax.set_title(split)
            ax.set_xlabel('x [m]')
            ax.set_aspect('equal', 'box')
            if i == 0:
                ax.set_ylabel('y [m]')
            ax.scatter(all_coords_plot[split][:,0], all_coords_plot[split][:,1], 
                       c = all_colours[split], s = all_sizes[split])
            for poly in POLY_DICT[split]:
                xy = np.array(poly.exterior.xy) \
                     - split_mean[split].T.reshape(-1,1)
                ax.plot(*xy, 'k-')
        
        # plt.tight_layout()
        plt.show()

    if args.query_requires_ground:
        ground_positives = "_ground-positives-required_"
    elif args.ground_aerial_positives_only:
        ground_positives = "_ground-aerial-only_"
    else:
        ground_positives = "_"
    train_file_baseline_basename = path.join(save_dir, f"training_queries_above-under_egonn_style_pos{args.pos_thresh:.0f}m_baseline{ground_positives}")
    # train_file_refined_basename = path.join(save_dir, f"training_queries_above-under_refined{ground_positives}")
    # test_file_base = path.join(save_dir, "test_queries_above-under_")
    # train_file_baseline_basename = path.join(save_dir, f"training_queries_QCAT_reloc_v2_")
    # test_file_base = path.join(save_dir, "test_queries_QCAT_reloc_v2_")
    construct_training_query_dict(df_train_baseline, train_file_baseline_basename)
    # construct_training_query_dict(df_train_refined, train_file_refined_basename)
    # construct_training_query_dict(df_test, test_file_base, test_set=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, required = True, 
                        help='Root directory containing splits of above_under dataset (ideally after postprocessing)')
    parser.add_argument('--save_dir', type = str, default = None, 
                        help='Directory to save training queries to, default is --root')
    parser.add_argument('--splits', nargs = '+', default = [], 
                        help='Splits (min 1) in root folder to process. Processes every folder in root if empty.')
    parser.add_argument('--exclude_runs', nargs = '+', default = [], 
                        help='Runs to exclude from training tuples.')
    # parser.add_argument('--radius_max', type = float, default = 30, 
    #                     help = 'Max radius (m) of submaps')
    parser.add_argument('--eval_thresh', type = float, required = True,
                        help = 'Threshold of nearest database submap for choosing eval queries')
    parser.add_argument('--pos_thresh', type = float, required = True, 
                        help = 'Threshold (m) for positive matches, default 0.5*radius')
    parser.add_argument('--neg_thresh', type = float, required = True, 
                        help = 'Threshold (m) for negative matches, default 2*radius')
    parser.add_argument('--buffer_thresh', type = float, required = True, 
                        help = 'Threshold (m) from ground positives to keep as buffer zone, default 2*radius')
    parser.add_argument('--icp', default = False, action = 'store_true', 
                        help = 'Use ICP refinement for GT pose')
    parser.add_argument('--cache_submaps', default = False, action = 'store_true', 
                        help = 'Cache submaps to speed up ICP (disable if RAM limited)')
    parser.add_argument('--query_requires_ground', default = False, action = 'store_true', 
                        help = 'Only save training queries that either are from the ground, or have at least 1 ground positive (to dissuade massive aerial bias). Can cause training issues for MinkLoc-based approaches (aerial negatives are never loaded into batches)')
    parser.add_argument('--ground_aerial_positives_only', default = False, action = 'store_true', 
                        help = 'Only save training queries and positives that contain ground/aerial matches. Removes ground/ground or aerial/aerial positives from training.')
    parser.add_argument('--viz', default = False, action = 'store_true', 
                        help = 'Enable visualisations of train/test splits')
    parser.add_argument('--debug', default = False, action = 'store_true', 
                        help = 'Debug mode, only process a handful of submaps per split')
    args = parser.parse_args()
    
    # if args.pos_thresh < 0:
    #     args.pos_thresh = 0.5 * args.radius_max
    # if args.neg_thresh < 0:
    #     args.neg_thresh = 2 * args.radius_max
    # if args.buffer_thresh < 0:
    #     args.buffer_thresh = 2 * args.radius_max
    if args.query_requires_ground and args.ground_aerial_positives_only:
        print("[WARNING] --ground_aerial_positives_only will supersede --query_requires_ground, thus the latter will have no effect")
        args.query_requires_ground = False
    
    args.save_dir = args.root if args.save_dir is None else args.save_dir
    
    print(args)
    main()
