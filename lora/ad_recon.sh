#! /bin/bash

class_name="cable_gland"
folder_name="2_1_res_64_up_16_up_good_1_anormal_8_bent_12"
lora_folder="epoch-000019.safetensors"
network_weight_folder="../result/MVTec3D-AD_experiment/${class_name}/lora_training/anormal/${folder_name}/models"

port_number=59999

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config \
  --main_process_port ${port_number} ad_recon.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --resolution '512,512' \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/${class_name} \
  --seed 42 \
  --network_weights ${network_weight_folder}  \
  --num_ddim_steps 30 \
  --trg_lora_epoch ${lora_folder} \
  --inner_iter 10 \
  --negative_prompt "low quality, worst quality, bad anatomy, bad composition, poor, low effort" \
  --guidance_scale 8.5 \
  --truncate_length 2 --cross_map_res [64] --trg_position "['up']" \
  --trg_part "attn_2"