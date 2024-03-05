#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --mem=160gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mail-user=ethan.griffiths@data61.csiro.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

module load cuda/11.1.1 gcc/9.3.0
source /home/gri317/mambaforge/etc/profile.d/conda.sh
conda activate ppt-net

sh train.sh pptnet_above-under_refined configs/pptnet_above-under_refined.yaml
