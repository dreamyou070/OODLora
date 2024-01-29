#! /bin/bash

class_name="cable_gland"
data_source='train_normal_small'
data_folder='MVTec3D-AD'
train_data_dir="../../../MyData/anomaly_detection/${data_folder}/${class_name}/${data_source}/rgb"
normal_folder='normal'
save_folder="res_32_up_only_normal_20240128_small_data"
output_dir="../result/${data_folder}_experiment/${class_name}/lora_training/${normal_folder}/${save_folder}"

start_epoch=0
port_number=50019

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config \
  --main_process_port $port_number ../lora/train_normal.py \
  --process_title parksooyeon \
  --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc  \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 --train_batch_size 1 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 --resolution '512,512' --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --sample_prompts ../../../MyData/anomaly_detection/inference.txt \
  --max_train_steps 160000 --attn_loss --task_loss_weight 1.0 --seed 42 --class_caption 'good' \
  --wandb_init_name ${class_name} \
  --train_data_dir "$train_data_dir" \
  --start_epoch $start_epoch \
  --output_dir "$output_dir" --truncate_pad --truncate_length 2  \
  --cross_map_res "[32]" --detail_64_up --trg_position "['up']" \
  --trg_part '["attn_1"]' --cls_training  # --normal_with_background #  --network_weights "$network_weights"