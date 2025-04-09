#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --mem=100gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1

##SBATCH --mail-user=ethan.griffiths@data61.csiro.au
##SBATCH --mail-type=FAIL
##SBATCH --mail-type=END

source ../../env-setup.sh
source /scratch3/gri317/venvs/hot-net/bin/activate

python generate_train_test_tuples.py \
	--root '/datasets/work/mlai-fsp-st-div/source/above-under/processed_submaps/hotnet_clouds/rad30m/postproc_voxel_ds_0.80m' \
	--radius_max '30' \
	--buffer_thresh '60' \
	--v2_only \
