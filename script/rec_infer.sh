#! /bin/bash

class_name="bagel"
folder_name="res_32_64_up_t_2_20240128_with_gaussian_noise"
data_folder="MVTec3D-AD"
normality_folder='normal'
network_weight_folder="../result/${data_folder}_experiment/${class_name}/lora_training/${normality_folder}/${folder_name}/models"
img_folder="../../../MyData/anomaly_detection/${data_folder}/${class_name}"

port_number=50801

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config \
  --main_process_port ${port_number} ../lora/rec_infer.py \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --sample_sampler ddim \
  --seed 42 \
  --concept_image_folder "${img_folder}" \
  --resolution '512,512' \
  --network_module networks.lora \
  --network_dim 64 \
  --network_alpha 4 \
  --network_weights ${network_weight_folder}  \
  --cross_map_res "[64,32]" \
  --trg_position "['up']" \
  --trg_part "['attn_0','attn_1','attn_2']" \
  --num_ddim_steps 50 \
  --prompt 'good' \
  --negative_prompt "low quality, worst quality, bad anatomy, bad composition, poor, low effort" \
  --guidance_scale 8.5 \
  --truncate_length 2 \
  --start_from_final \
  --only_zero_save \
  --use_pixel_mask \
  --class_name ${class_name} \
  --anormal_thred 0.5 \
  --only_zero_save --only_normal_infer