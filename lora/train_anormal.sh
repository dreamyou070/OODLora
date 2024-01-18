#!bin/bash

class_name="cable_gland"
data_source='train_ex'
train_data_dir="../../../MyData/anomaly_detection/MVTec3D-AD/${class_name}/${data_source}/rgb"
normal_folder='anormal'
save_folder="2_1_res_64_up_16_up_good_1_anormal_8_bent_12"
output_dir="../result/MVTec3D-AD_experiment/${class_name}/lora_training/${normal_folder}/${save_folder}"
network_weights="${output_dir}/models/epoch-000019.safetensors"
port_number=54164
start_epoch=0

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config --main_process_port $port_number train_anormal.py \
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
  --wandb_init_name "$class_name" \
  --train_data_dir "$train_data_dir" \
  --start_epoch $start_epoch \
  --output_dir "$output_dir" \
  --cross_map_res [64] --detail_64_up --trg_position "['up']" \
  --trg_part '["attn_2","attn_1","attn_0"]' --truncate_pad --truncate_length 3 --cls_training \
  --normal_with_background --network_weights "$network_weights"