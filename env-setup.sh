#!/bin/bash
module load python/3.12.0 pytorch/2.1.1-py312-cu122-mpi open3d/0.18.1
export PYTHONPATH=$PYTHONPATH:'/scratch3/gri317/repos/HOT-Net'
export WANDB_ENTITY='ethan-phd'