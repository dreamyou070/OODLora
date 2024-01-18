#! /bin/bash

class_name="carrot"
folder_name="2_1_res_64_up_16_up_1_good_8_anormal"
lora_folder="epoch-000014.safetensors"
network_weight_folder="../result/MVTec3D-AD_experiment/${class_name}/lora_training/anormal/${folder_name}/models"

port_number=50518

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
  --num_ddim_steps 50 \
  --trg_lora_epoch ${lora_folder} \
  --inner_iter 10 \
  --only_zero_save \
  --truncate_length 2 --cross_map_res [16] --trg_position "['up']" \
  --use_avg_mask
  #--trg_part "attn_2" --use_avg_mask