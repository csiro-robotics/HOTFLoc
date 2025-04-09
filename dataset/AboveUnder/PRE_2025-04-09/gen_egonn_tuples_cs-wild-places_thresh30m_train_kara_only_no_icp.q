#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --mem=200gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1

#SBATCH --mail-user=ethan.griffiths@data61.csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

# NOTE: run this on petrichor

source /home/gri317/miniforge3/etc/profile.d/conda.sh
conda activate above-under

python -c "import open3d as o3d; print('Import success')"

# python generate_train_test_tuples_egonn_style.py \
# 	--root '/datasets/work/d61-cps-slam/source/CS-Wild-Places/downsampled_voxel_0.80m' \
# 	--save_dir "/scratch3/gri317/repos/HOT-Net/dataset/AboveUnder/tuples_egonn_thresh30m_train_kara_only" \
# 	--eval_thresh '30' \
# 	--pos_thresh "15" \
# 	--neg_thresh "60" \
# 	--buffer_thresh '30' \
# 	--ground_aerial_positives_only \
# 	# --query_requires_ground \

python generate_train_test_tuples_egonn_style.py \
	--root '/datasets/work/d61-cps-slam/source/CS-Wild-Places/downsampled_voxel_0.80m' \
	--save_dir "/scratch3/gri317/repos/HOT-Net/dataset/AboveUnder/tuples_egonn_thresh30m_train_kara_only" \
	--eval_thresh '30' \
	--pos_thresh "15" \
	--neg_thresh "60" \
	--buffer_thresh '30' \
