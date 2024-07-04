#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --mem=300gb
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
	--config '../config/config_baseline_octf_depth9_lr1e-4_sched70_350.txt' \
	--model_config '../models/hotformer_4stage_2-18-2-2_ctprop_layerscale1e-5_cfg.txt' \
