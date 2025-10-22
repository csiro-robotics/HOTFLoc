#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mail-user=ethan.griffiths@data61.csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

cd /scratch3/gri317/repos/HOT-Net
source env-setup.sh
source /scratch3/gri317/venvs/hotformerloc-geotransformer/bin/activate
export OMP_NUM_THREADS=6

cd misc/
python runtime_eval_metloc_rerank_real_data.py \
    --config ../config/config_cs-wild-places_voxel0.4m_egonn_noicp.txt \
	--model_config ../models/cfg_files/egonn.txt  \
	--precompute_descriptors \
	--num_neighbours 20 \
	--use_sgv \
	&> runtime_experiments/CSWP_egonn_sgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/config_wild-places_egonn.txt \
	--model_config ../models/cfg_files/egonn.txt  \
	--precompute_descriptors \
	--num_neighbours 20 \
	--use_sgv \
	&> runtime_experiments/WP_egonn_sgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/config_mulran_egonn_twostageicp.txt \
	--model_config ../models/cfg_files/egonn.txt  \
	--precompute_descriptors \
	--num_neighbours 20 \
	--use_sgv \
	&> runtime_experiments/mulran_egonn_sgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/exp_reranking/config_metloc_rr_cs-wild-places_d7_tripletbs256_mesa_augmode4_radius1.6_noicp_rrbs32_finetune.txt \
	--model_config ../models/cfg_files/reranking/hotformermetricloc_cart_ch192_pyrmd_2lvls_featembed128_radius1.6_3stage_metloc_rr_gc5.0_5.0_5.0_noOT_cfg.txt \
	--precompute_descriptors \
	--num_neighbours 20 \
	--use_sgv \
	&> runtime_experiments/CSWP_HOTFLocSmall_sgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/exp_reranking/config_metloc_rr_cs-wild-places_d7_tripletbs256_mesa_augmode4_radius1.6_noicp_rrbs32_finetune.txt \
	--model_config ../models/cfg_files/reranking/hotformermetricloc_cart_ch192_pyrmd_2lvls_featembed128_radius1.6_3stage_metloc_rr_gc5.0_5.0_5.0_noOT_cfg.txt \
	--precompute_descriptors \
	--num_neighbours 20 \
	&> runtime_experiments/CSWP_HOTFLocSmall_msgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/exp_reranking/config_metloc_rr_cs-wild-places_d7_tripletbs256_mesa_augmode4_radius1.6_noicp_rrbs32_finetune.txt \
	--model_config ../models/cfg_files/reranking/hotformermetricloc_cart_ch256_3lvls_up4fine_up2coarse_featembed128_radius1.6_3stage_metloc_rr_gc5.0_5.0_5.0_noOT_FIXED_cfg.txt \
	--precompute_descriptors \
	--num_neighbours 20 \
	&> runtime_experiments/CSWP_HOTFLocBase_msgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/exp_reranking/config_metloc_rr_wild-places_d7_tripletbs256_mesa_augmode4_radius1.6_rrbs64_finetune.txt \
	--model_config ../models/cfg_files/reranking/hotformermetricloc_cyl_ch192_pyrmd_2lvls_featembed128_radius1.6_3stage_metloc_rr_gc5.0_5.0_5.0_noOT_cfg.txt \
	--precompute_descriptors \
	--num_neighbours 20 \
	--use_sgv \
	&> runtime_experiments/WP_HOTFLocSmall_sgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/exp_reranking/config_metloc_rr_wild-places_d7_tripletbs256_mesa_augmode4_radius1.6_rrbs64_finetune.txt \
	--model_config ../models/cfg_files/reranking/hotformermetricloc_cyl_ch192_pyrmd_2lvls_featembed128_radius1.6_3stage_metloc_rr_gc5.0_5.0_5.0_noOT_cfg.txt \
	--precompute_descriptors \
	--num_neighbours 20 \
	&> runtime_experiments/WP_HOTFLocSmall_msgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/exp_reranking/config_metloc_rr_wild-places_d7_tripletbs256_mesa_augmode4_radius1.6_rrbs64_finetune.txt \
	--model_config ../models/cfg_files/reranking/hotformermetricloc_cyl_ch256_3lvls_featembed128_radius1.6_3stage_metloc_rr_gc5.0_3.2_1.6_noOT_cfg \
	--precompute_descriptors \
	--num_neighbours 20 \
	&> runtime_experiments/WP_HOTFLocBase_msgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/exp_reranking/config_metloc_rr_mulran_d9_tripletbs256_mesa_augmode4_twostageicp_rrbs128_finetune.txt \
	--model_config ../models/cfg_files/reranking/hotformermetricloc_cart_ch192_pyrmd_2lvls_featembed128_radius0.6_3stage_metloc_rr_gc5.0_5.0_5.0_noOT_cfg.txt \
	--precompute_descriptors \
	--num_neighbours 20 \
	--use_sgv \
	&> runtime_experiments/mulran_HOTFLocSmall_sgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/exp_reranking/config_metloc_rr_mulran_d9_tripletbs256_mesa_augmode4_twostageicp_rrbs128_finetune.txt \
	--model_config ../models/cfg_files/reranking/hotformermetricloc_cart_ch192_pyrmd_2lvls_featembed128_radius0.6_3stage_metloc_rr_gc5.0_5.0_5.0_noOT_cfg.txt \
	--precompute_descriptors \
	--num_neighbours 20 \
	&> runtime_experiments/mulran_HOTFLocSmall_msgv.out

python runtime_eval_metloc_rerank_real_data.py \
	--config ../config/exp_reranking/config_metloc_rr_mulran_d9_tripletbs256_mesa_augmode4_twostageicp_rrbs128_finetune.txt \
	--model_config ../models/cfg_files/reranking/hotformermetricloc_cart_ch256_3lvls_up4fine_featembed128_radius0.6_3stagecoarse_metloc_rr_gc5.0_5.0_5.0_noOT_cfg.txt \
	--precompute_descriptors \
	--num_neighbours 20 \
	&> runtime_experiments/mulran_HOTFLocBase_msgv.out

exit