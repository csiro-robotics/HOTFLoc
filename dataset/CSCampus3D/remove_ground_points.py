"""
Script to remove ground points from campus3d submaps. 
"""
import os
import glob

import numpy as np
import CSF
import submitit

from dataset.pointnetvlad.pnv_raw import PNVPointCloudLoader
from misc.point_clouds import plot_points

MIN_POINTS = 50
CSF_RIGIDNESS = 2
CSF_THRESHOLD = 1.0       # m from cloth to classify as ground
CSF_RESOLUTION = 1.0         # m
CSF_BSLOOPSMOOTH = True
NUM_THREADS = 16

def remove_ground_CSF(pts: np.ndarray, debug=True) -> np.ndarray:
    """
    Remove ground points using the Cloth Simulation Filter method.
    """
    csf = CSF.CSF()
    csf.params.bSloopSmooth = CSF_BSLOOPSMOOTH
    csf.params.cloth_resolution = CSF_RESOLUTION
    csf.params.rigidness = CSF_RIGIDNESS
    csf.params.threshold = CSF_THRESHOLD

    # NOTE: Due to precision issues, the point cloud must first be resized to a
    #       larger range
    pts_resized = pts * 50  # allegedly, the original clouds were 100m across
    csf.setPointCloud(pts_resized)
    ground = CSF.VecInt()       # index of ground points
    non_ground = CSF.VecInt()   # index of non-ground points
    csf.do_filtering(ground, non_ground, exportCloth=False)
    
    if len(np.array(non_ground)) > 0:
        filtered_pts = pts[np.array(non_ground)] # extract non-ground points
    else:
        filtered_pts = np.array([])  # handle case where all pts are ground
    # if debug:
        # print(len(np.array(non_ground)))
        # if len(filtered_pts > 0):
        #     plot_points(pts_resized)
        #     plot_points(filtered_pts)
    return filtered_pts

def save_pc(cloud: np.ndarray, cloud_save_path: str):
    """Save pointcloud in original format."""
    cloud = np.float64(cloud)
    cloud = np.reshape(cloud, cloud.size)
    with open(cloud_save_path, 'wb') as f:
        cloud.tofile(f)
    return None

def remove_ground_in_subdir(cloud_subdir: str):
    """
    Remove ground points with CSF for all point clouds in subdir.
    """
    assert os.path.exists(cloud_subdir), "Subdirectory not found"
    cloud_paths = glob.glob(f"{cloud_subdir}/**/*.bin", recursive=True)
    assert len(cloud_paths) > 0, "No valid .bin pcls found"

    loader = PNVPointCloudLoader()
    num_skipped = 0
    for path in cloud_paths:
        cloud = loader.read_pc(path)
        cloud_rmground = remove_ground_CSF(cloud)
        if len(cloud_rmground) >= MIN_POINTS:
            cloud_rel_path = str.split(path, ROOT_PATH)[-1]
            cloud_save_path = os.path.join(SAVE_PATH, cloud_rel_path)
            cloud_save_dir = os.path.split(cloud_save_path)[0]
            if not os.path.exists(cloud_save_dir):
                os.makedirs(cloud_save_dir)
            save_pc(cloud_rmground, cloud_save_path)
            continue
        num_skipped += 1
    print(f"Num skipped: {num_skipped}")
    return None            

if __name__ == "__main__":
    global ROOT_PATH, SAVE_PATH
    ROOT_PATH = "/scratch3/gri317/datasets/cs_campus3d/benchmark_datasets/umd/" # NOTE: MUST INCLUDE THE FINAL SLASH FOR NOW
    SAVE_PATH = "/scratch3/gri317/datasets/cs_campus3d/benchmark_datasets_rmground/umd/"
    log_folder = "/scratch3/gri317/repos/HOT-Net/dataset/campus3d/submitit_logs"
    assert os.path.exists(ROOT_PATH)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    # Load all subfolders and submit those as jobs
    subdir_list = sorted([path for path in glob.glob(f"{ROOT_PATH}/*") if os.path.isdir(path)])

    executor = submitit.AutoExecutor(folder=log_folder)
    # executor = submitit.AutoExecutor(folder=log_folder, cluster='debug')
    
    executor.update_parameters(name="campus3d_rmground", timeout_min=int(2*24*60),
                               nodes=1, cpus_per_task=NUM_THREADS, tasks_per_node=1,
                               slurm_mem="16gb",
                               slurm_mail_user="ethan.griffiths@data61.csiro.au",
                               slurm_mail_type="FAIL")
    print("Submitting jobs")
    jobs = executor.map_array(remove_ground_in_subdir, subdir_list)
    
    outputs = [job.result() for job in jobs]
    print(outputs)
    print("All jobs complete")
