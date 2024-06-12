#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --mem=200gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mail-user=ethan.griffiths@data61.csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

source env-setup.sh
source /scratch3/gri317/venvs/hot-net/bin/activate

cd training/
python train.py \
	--config '../config/config_refined_octf_depth9_lr5e-4_sched80_350.txt' \
	--model_config '../models/octformer_4stage_18blocks_2ndstage_2ds_2topdown_cfg.txt' \
