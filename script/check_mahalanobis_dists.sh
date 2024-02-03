#! /bin/bash
class_name="bagel"
data_source='train_normal'
data_folder='MVTec3D-AD'
train_data_dir="../../../MyData/anomaly_detection/${data_folder}/${class_name}/${data_source}/rgb"
all_data_dir="../../../MyData/anomaly_detection/${data_folder}/${class_name}/train_ex2/rgb"
normal_folder='normal'
save_folder="res_64_down_task_loss_mahal_dist_attn_loss_0.001"
output_dir="../result/${data_folder}_experiment/${class_name}/lora_training/${normal_folder}/${save_folder}"
network_weights="../result/${data_folder}_experiment/${class_name}/lora_training/${normal_folder}/res_64_down_task_loss_mahal_dist_attn_loss_0.001/models/epoch-000006.safetensors"

start_epoch=0
port_number=59517

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config \
  --main_process_port $port_number ../lora/check_mahalanobis_dists.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --logging_dir "${output_dir}/logs" \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --resolution '512,512' \
  --class_caption 'good' \
  --all_data_dir "${all_data_dir}" \
  --output_dir "$output_dir" \
  --do_check_anormal \
  --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
  --cls_training \
  --network_weights "$network_weights"