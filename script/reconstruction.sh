#! /bin/bash

class_name="bagel"
folder_name="res_64_up_2_task_loss_mahal_dist_attn_loss_0.008_actdeact_mahal_anomal"
data_name="MVTec3D-AD"
normality_folder='normal'
network_weight_folder="../result/${data_name}_experiment/${class_name}/lora_training/${normality_folder}/${folder_name}/models"
detection_network_weights="../result/${data_name}_experiment/${class_name}/lora_training/${normality_folder}/object_detection/models/epoch-000110.safetensors"
img_folder="../../../MyData/anomaly_detection/${data_name}/${class_name}"
port_number=50015

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port ${port_number} ../lora/reconstruction.py \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --sample_sampler ddim \
  --seed 42 \
  --concept_image_folder "${img_folder}" \
  --resolution '512,512' \
  --network_module networks.lora \
  --network_dim 64 \
  --network_alpha 4 \
  --network_weights ${network_weight_folder} \
  --detection_network_weights ${detection_network_weights} \
  --num_ddim_steps 30 \
  --prompt 'good' \
  --negative_prompt "low quality, worst quality, bad anatomy, bad composition, poor, low effort" \
  --guidance_scale 8.5 \
  --truncate_length 2 \
  --only_zero_save \
  --class_name ${class_name} \
  --anormal_thred 0.5 \
  --trg_layer "up_blocks_3_attentions_2_transformer_blocks_0_attn2" \
  --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']"