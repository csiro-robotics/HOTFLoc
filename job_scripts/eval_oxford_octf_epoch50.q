#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mail-user=ethan.griffiths@data61.csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

source env-setup.sh
source /scratch3/gri317/venvs/hot-net/bin/activate

cd eval/
python pnv_evaluate.py \
	--config '../config/config_baseline_octf.txt' \
	--model_config '../models/octformer_cfg_ds.txt' \
	--weights '../weights/Oxford/OctFormer_20240507_1545_50.pth' \
