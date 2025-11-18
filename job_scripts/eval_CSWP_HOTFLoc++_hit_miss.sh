#!/bin/bash
#SBATCH --time=0-06:00:00
#SBATCH --mem=300gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mail-user=ethan.griffiths@data61.csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

source env-setup.sh
source /scratch3/gri317/venvs/hotformerloc-geotransformer/bin/activate

cd eval/
python evaluate_metric_loc_splits_rerank_hit_miss.py \
    --config ../config/config_hotfloc++_cs-wild-places_voxel0.4m_stage2.txt \
    --model_config ../models/cfg_files/hotfloc++_cs-wild-places_cfg.txt \
    --weights ../weights/hotfloc++_cs-wild-places_epoch60.ckpt \
    --radius 10 30 \
    --only_global \
    # --save_embeddings \
