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
	--config '../config/config_AboveUnder_baseline_octf_20k.txt' \
	--model_config '../models/minkloc3dv2.txt' \
	--weights '../weights/AboveUnder_baseline/MinkLoc_20240507_1639_final.pth' \
