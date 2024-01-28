#! /bin/bash

class_name="bagel"
data_source='train_normal'
data_folder='MVTec3D-AD'
train_data_dir="../../../MyData/anomaly_detection/${data_folder}/${class_name}/${data_source}/rgb"
normal_folder='normal'
save_folder="res_64_up_attn_2_20240128_with_random_position_shuffling"
output_dir="../result/${data_folder}_experiment/${class_name}/lora_training/${normal_folder}/${save_folder}"
#network_weights="../result/${data_folder}_experiment/${class_name}/lora_training/${normal_folder}/res_64_up_attn2_t_2_attn2/models/epoch-000007.safetensors"
start_epoch=0
port_number=59001


NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
  --main_process_port $port_number ../lora/train_anormal_with_noise.py \
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
  --cross_map_res "[64]" --detail_64_up --trg_position "['up']" --normal_with_background \
  --trg_part '["attn_2"]' --cls_training --all_same_learning # --network_weights "$network_weights"
