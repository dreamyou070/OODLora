#! /bin/bash
class_name="bagel"
data_source='train_normal'
data_folder='MVTec3D-AD'
all_data_dir="../../../MyData/anomaly_detection/${data_folder}/${class_name}/train_ex/rgb"
normal_folder='normal'
save_folder="random_vector_generating"
output_dir="../result/${data_folder}_experiment/${class_name}/lora_training/${normal_folder}/${save_folder}"
network_weights="../result/${data_folder}_experiment/${class_name}/lora_training/${normal_folder}/res_64_down_1_task_loss_mahal_dist_attn_loss_0.008_actdeact_mahal_anomal/models/epoch-000001.safetensors"

start_epoch=0
port_number=59518

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config \
  --main_process_port $port_number ../lora/random_vector_generation.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --logging_dir "${output_dir}/logs" \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --seed 42 \
  --resolution '512,512' \
  --class_caption 'good' \
  --all_data_dir "${all_data_dir}" \
  --output_dir "$output_dir" \
  --truncate_length 2 \
  --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2']" \
  --cls_training \
  --do_check_anormal \
  --network_weights "$network_weights"