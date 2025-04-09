#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mail-user=ethan.griffiths@data61.csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

source env-setup.sh
source /scratch3/gri317/venvs/hot-net/bin/activate

cd eval/
python pnv_evaluate.py \
	--config '../config/config_CS-WildPlaces_rmground_thresh30m_stdnorm.txt' \
	--model_config '../models/minkloc3dv2.txt' \
	--weights '../weights/AboveUnder_voxel0.8m_baseline/MinkLoc_20240823_1515_job57089552_e200.pth' 
	
	# eval model trained on Kara, on new data
