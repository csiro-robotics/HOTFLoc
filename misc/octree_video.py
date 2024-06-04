"""
Generate video of Octrees.

Written by Ethan Griffiths (Data61, Pullenvale)
"""
import argparse 
import open3d as o3d
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
from ocnn.octree import Octree, Points

from dataset.pointnetvlad.pnv_raw import PNVPointCloudLoader
from dataset.AboveUnder.AboveUnder_raw import AboveUnderPointCloudLoader
from misc.utils import octree_to_points

def main():
    no_vis = args.no_vis
    binPointCloudLoader = PNVPointCloudLoader()
    pcdPointCloudLoader = AboveUnderPointCloudLoader()
    clouds = sorted(glob(f"{args.clouds_path}/*.pcd") + glob(f"{args.clouds_path}/*.bin"))
    assert len(clouds) > 0, "No valid point cloud files found"
    
    stats_original = {'mean' : [], 'min' : [], 'max' : [], 'size' : []}
    stats_octree = {'mean' : [], 'min' : [], 'max' : [], 'size' : []}
    
    if not no_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # geometry is the point cloud used in your animaiton
        original_cloud_vis = o3d.geometry.PointCloud()
        octree_cloud_vis = o3d.geometry.PointCloud()
    
    tic = None
    for idx, cloud_path in tqdm(enumerate(clouds), total=len(clouds)):
        tic = time.time()
        if os.path.splitext(cloud_path)[-1] == ".bin":
            points_original = binPointCloudLoader.read_pc(cloud_path)
        elif os.path.splitext(cloud_path)[-1] == ".pcd":
            points_original = pcdPointCloudLoader.read_pc(cloud_path)
        else:
            raise ValueError('Invalid point cloud type, must be .bin or .pcd')

        points_tensor = torch.tensor(points_original, dtype=torch.float)
        # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
        points_tensor = torch.clamp(points_tensor, -1, 1)
        # Convert to ocnn Points object, then create Octree
        points_ocnn = Points(points_tensor)
        octree = Octree(args.octree_depth, full_depth=2)
        octree.build_octree(points_ocnn)
        # Convert back to points
        points_octree = octree_to_points(octree).numpy()        

        # Convert to o3d point cloud
        original_cloud = o3d.geometry.PointCloud()
        original_cloud.points = o3d.utility.Vector3dVector(points_original)
        octree_cloud = o3d.geometry.PointCloud()
        octree_cloud.points = o3d.utility.Vector3dVector(points_octree)
        
        # Stat tracking
        stats_original['mean'].append(np.mean(np.array(original_cloud.points), 0))
        stats_original['min'].append(np.min(np.array(original_cloud.points), 0))
        stats_original['max'].append(np.max(np.array(original_cloud.points), 0))
        stats_original['size'].append(len(original_cloud.points))
        stats_octree['mean'].append(np.mean(np.array(octree_cloud.points), 0))
        stats_octree['min'].append(np.min(np.array(octree_cloud.points), 0))
        stats_octree['max'].append(np.max(np.array(octree_cloud.points), 0))
        stats_octree['size'].append(len(octree_cloud.points))

        # Shift Octree cloud so it does not overlap with original cloud
        tf_offset = np.eye(4)
        x_shift = 2.2
        tf_offset[0,3] = x_shift
        octree_cloud.transform(tf_offset)
        
        if not no_vis:
            original_cloud_vis.points = original_cloud.points
            octree_cloud_vis.points = octree_cloud.points
            if idx == 0:
                vis.add_geometry(original_cloud_vis)
                vis.add_geometry(octree_cloud_vis)
            else:
                vis.update_geometry(original_cloud_vis)
                vis.update_geometry(octree_cloud_vis)
            vis.poll_events()
            vis.update_renderer()            
            
            # Wait
            if idx == 0 and args.delay_start > 0:   # wait for 5 seconds before playing
                while (time.time()-tic < args.delay_start):
                    vis.poll_events()
                    vis.update_renderer()
                
            while (time.time()-tic < args.update_delay):
                vis.poll_events()
                vis.update_renderer()
    
    print(f"Submaps average stats:\n"
          f"ORIGINAL:\n"
          f"Mean (x,y,z): {np.mean(stats_original['mean'], 0)}\n"
          f"Min (x,y,z):  {np.mean(stats_original['min'], 0)}\n"
          f"Max (x,y,z):  {np.mean(stats_original['max'], 0)}\n"
          f"Size:  mean - {np.mean(stats_original['size'], 0) if stats_original['size'] != [] else 'NaN'}, \
              min - {np.min(stats_original['size'], 0) if stats_original['size'] != [] else 'NaN'}, \
                  max - {np.max(stats_original['size'], 0) if stats_original['size'] != [] else 'NaN'}\n"
          f"OCTREE DEPTH {args.octree_depth}:\n"
          f"Mean (x,y,z): {np.mean(stats_octree['mean'], 0)}\n"
          f"Min (x,y,z):  {np.mean(stats_octree['min'], 0)}\n"
          f"Max (x,y,z):  {np.mean(stats_octree['max'], 0)}\n"
          f"Size:  mean - {np.mean(stats_octree['size'], 0) if stats_octree['size'] != [] else 'NaN'}, \
              min - {np.min(stats_octree['size'], 0) if stats_octree['size'] != [] else 'NaN'}, \
                  max - {np.max(stats_octree['size'], 0) if stats_octree['size'] != [] else 'NaN'}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clouds_path', type = str, required=True, help="path to processed submaps")
    parser.add_argument('--octree_depth', type = int, required=True, help="depth of octree to construct")
    parser.add_argument('--no_vis', action = 'store_true', help="no visualisations, only print submap stats")
    parser.add_argument('--update_delay', type = float, default=0.1, help="time to wait between frames")
    parser.add_argument('--delay_start', type=float, default=0, help="time (s) to wait before starting, to setup viewer and screen recorder")
    args = parser.parse_args()
    assert args.octree_depth > 0, 'Octree depth must be positive'
    main()