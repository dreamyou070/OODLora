#!/bin/bash

# start of the script
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0


python data_check.py --device cuda \
    --data_folder ../../../../MyData/anomaly_detection/MVTec