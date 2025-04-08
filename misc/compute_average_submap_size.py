"""
Compute the average number of points per submap for ground and aerial submaps.
By Ethan Griffiths (Data61, Pullenvale).
"""
import os
import glob
import open3d as o3d
from tqdm import tqdm

def main():
    # Find all ground submaps
    data_root = '/scratch3/gri317/datasets/CS-Wild-Places/postproc_voxel_0.80m_rmground'

    forests = os.listdir(data_root)
    total_avg = dict.fromkeys(['ground', 'aerial'], 0)
    total_count = dict.fromkeys(['ground', 'aerial'], 0)
    for forest in forests:
        forest_avg = dict.fromkeys(['ground', 'aerial'], 0)
        forest_count = dict.fromkeys(['ground', 'aerial'], 0)
        print(f'{forest}:')
        forest_path = os.path.join(data_root, forest)
        ground_submaps = glob.glob('*ground*/**/*.pcd', root_dir=forest_path, recursive=True)
        aerial_submaps = glob.glob('*aerial*/**/*.pcd', root_dir=forest_path, recursive=True)
        for ground_submap in tqdm(ground_submaps):
            pc_o3d = o3d.io.read_point_cloud(os.path.join(forest_path, ground_submap))
            forest_avg['ground'] += len(pc_o3d.points)
            forest_count['ground'] += 1
        for aerial_submap in tqdm(aerial_submaps):
            pc_o3d = o3d.io.read_point_cloud(os.path.join(forest_path, aerial_submap))
            forest_avg['aerial'] += len(pc_o3d.points)
            forest_count['aerial'] += 1
        for split_type in forest_avg.keys():
            total_avg[split_type] += forest_avg[split_type]
            total_count[split_type] += forest_count[split_type]
        print(f'  Ground avg: {forest_avg['ground'] / forest_count['ground']}')
        print(f'  Aerial avg: {forest_avg['aerial'] / forest_count['aerial']}')
        
    print(f'Overall Ground avg: {total_avg['ground'] / total_count['ground']}')
    print(f'Overall Aerial avg: {total_avg['aerial'] / total_count['aerial']}')
    total_avg_both = ((total_avg['ground'] + total_avg['aerial'])
                          / (total_count['ground'] + total_count['aerial']))
    print(f'Total avg: {total_avg_both}')

if __name__ == '__main__':
    main()