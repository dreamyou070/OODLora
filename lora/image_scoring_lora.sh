#!/bin/bash

# start of the script
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

source ~/.bashrc
conda activate venv
ml purge
ml load cuda/11.0


python image_scoring_lora.py --device cuda \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 \
  --network_weights /data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/3_contrastive_learning_eps_0.0_increase_generality/epoch-000003.safetensors