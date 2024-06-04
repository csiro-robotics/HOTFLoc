#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mail-user=ethan.griffiths@data61.csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

source /home/gri317/mambaforge/etc/profile.d/conda.sh
conda activate egonn-env

export PYTHONPATH=$PYTHONPATH:$(pwd)
cd training/
python train.py \
	--config '../config/config_AboveUnder_rmground.txt' \
	--model_config '../models/minkloc3dv2.txt' \
