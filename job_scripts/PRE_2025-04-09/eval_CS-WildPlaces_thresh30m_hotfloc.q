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
	--config '../config/config_CS-WildPlaces_thresh30m_stdnorm_octf_depth7_lr8e-4_sched50_mesa1.txt' \
	--model_config '../models/hotformerloc_best_CS-Wild-Places_cfg.txt' \
	--weights '../weights/AboveUnder_voxel0.8m_baseline/HOTFormerLoc-3Level-10Blocks-WindowSize64-ADaPE-PyramidAttnPoolMixer-k74-36-18_20241021_1735_job66202369_e50.pth' 
	
	# eval model trained on Kara, on new data
