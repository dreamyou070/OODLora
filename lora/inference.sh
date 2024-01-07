#!/bin/bash

# srun -p suma_a6000 -q big_qos --gres=gpu:2 --pty bash -i

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 56464 inference.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 \
  --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --resolution '512,512' \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/cookie \
  --seed 42 \
  --network_weights ../result/MVTec3D-AD_experiment/cookie/lora_training/res_64_up/models_before \
  --num_ddim_steps 50 \
  --cross_map_res [64] \
  --trg_position "['up']"