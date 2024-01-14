#!bin/bash

class_name="foam"
data_source='train_ex'
train_data_dir="../../../MyData/anomaly_detection/MVTec3D-AD/${class_name}/${data_source}/rgb"

save_folder="check_res_64_up_down_res_16_up_down_text_3_no_back_no_cls"
output_dir="../result/MVTec3D-AD_experiment/${class_name}/lora_training/anormal/${save_folder}"

port_number=54545
start_epoch=0

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_config --main_process_port $port_number train.py \
  --process_title parksooyeon \
  --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc  \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 --train_batch_size 1 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 --resolution '512,512' --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --sample_prompts ../../../MyData/anomaly_detection/inference.txt \
  --max_train_steps 160000 --use_attn_loss --task_loss_weight 1.0 --seed 42 --class_caption 'good' --start_epoch 0 \
  --wandb_init_name "$class_name" \
  --train_data_dir "$train_data_dir" \
  --start_epoch $start_epoch \
  --output_dir "$output_dir" \
  --cross_map_res [64,16] \
  --trg_position "['up','down']" \
  --trg_part '["attn_2","attn_1","attn_0"]' \
  --anormal_sample_normal_loss \
  --truncate_pad --truncate_length 3