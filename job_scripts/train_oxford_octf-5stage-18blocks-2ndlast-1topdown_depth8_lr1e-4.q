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
	--config '../config/config_baseline_octf_depth8_lr1e-4.txt' \
	--model_config '../models/octformer_5stage_18blocks_2ndlast_1topdown_cfg.txt' \
