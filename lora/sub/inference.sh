#!/bin/bash

class_name='bagel'
start_epoch=0
port_number=53102
folder_name="2_2_res_64_up_t_2"
concept_image_folder="../../../MyData/anomaly_detection/MVTec3D-AD/${class_name}"
output_dir="../result/MVTec3D-AD_experiment/${class_name}/lora_training/anormal/${folder_name}"
network_weights="${output_dir}/models"

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config \
  --main_process_port ${port_number} ../lora/inference.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --prompt 'good' --sample_sampler ddim \
  --resolution '512,512' --seed 42 \
  --cross_map_res [64] \
  --trg_position "['up']" \
  --concept_image_folder "${concept_image_folder}" \
  --network_weights "${network_weights}" \
  --truncate_length 2 \
  --start_epoch ${start_epoch} --trg_part "['attn_2']"
