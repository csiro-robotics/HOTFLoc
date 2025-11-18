#!/bin/bash
source env-setup.sh
source /scratch3/gri317/venvs/hotformerloc-geotransformer/bin/activate

cd eval/

python visualise_correspondences_single_eval_split_rerank_hit_miss_format.py \
    --config ../config/config_hotfloc++_cs-wild-places_voxel0.4m_stage2.txt \
    --model_config ../models/cfg_files/hotfloc++_cs-wild-places_cfg.txt \
    --weights ../weights/hotfloc++_cs-wild-places_epoch60.ckpt \
    --dataset_idx 1 \
    --database_split_idx 4 \
    --query_split_idx 3 \
    --save_dir ./vis_corr_eval_splits_rerank_hit_miss \
    --disable_animation \
    --non_interactive \
    --zoom 0.65 \
    --voxel_size 0.80 \
    --load_embeddings \
    --num_workers 2 \
    --verbose \
    --disable_reranking &

python visualise_correspondences_single_eval_split_rerank_hit_miss_format.py \
    --config ../config/config_hotfloc++_cs-wild-places_voxel0.4m_stage2.txt \
    --model_config ../models/cfg_files/hotfloc++_cs-wild-places_cfg.txt \
    --weights ../weights/hotfloc++_cs-wild-places_epoch60.ckpt \
    --dataset_idx 1 \
    --database_split_idx 4 \
    --query_split_idx 3 \
    --save_dir ./vis_corr_eval_splits_rerank_hit_miss \
    --disable_animation \
    --non_interactive \
    --zoom 0.65 \
    --voxel_size 0.80 \
    --load_embeddings \
    --num_workers 2 \
    --verbose &

wait