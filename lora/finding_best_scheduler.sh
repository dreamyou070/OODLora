#!/bin/bash

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

source ~/.bashrc
conda activate venv
ml purge
ml load cuda/11.0

python finding_best_scheduler.py  --device cuda \
  --process_title parksooyeon \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --prompt 'good' --sample_sampler ddim --num_ddim_steps 50 \
  --output_dir '/data7/sooyeon/Lora/OODLora/result' \
  --network_weights ../result/MVTec_experiment/bagel/2_contrastive_learning_eps_0.0_new_code_highrepeat/epoch-000002.safetensors \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTecAD/bagel




