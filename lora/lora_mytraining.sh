#!/bin/bash
# dreamyou070
# qkrtndus0701?!
# srun -p suma_a6000 -q big_qos --gres=gpu:2 --job-name=lora_9 --time=48:00:00 --pty bash -i
# conda activate venv_lora
# cd ./Lora/OODLora/lora
NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 51389 lora_mytraining.py \
  --process_title parksooyeon --max_token_length 225 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train/bad \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test/bad --seed 42 --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --logging_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/3_lora_normal_trining_32_res_attnmap/logs --normal_training \
  --wandb_run_name 3_lora_normal_trining_32_res_attnmap --cross_map_res [32] \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/3_lora_normal_trining_32_res_attnmap

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 51889 lora_mytraining.py \
  --process_title parksooyeon --max_token_length 225 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train/bad \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test/bad --seed 42 --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --logging_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/8_lora_diff_trining_32_res_attnmap/logs \
  --wandb_run_name 8_lora_diff_trining_32_res_attnmap --cross_map_res [32] \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/8_lora_diff_trining_32_res_attnmap

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 51489 lora_mytraining.py \
  --process_title parksooyeon --max_token_length 225 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train/bad \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test/bad --seed 42 --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --logging_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/4_lora_normal_trining_16_res_attnmap/logs --normal_training \
  --wandb_run_name 4_lora_normal_trining_16_res_attnmap --cross_map_res [16] \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/4_lora_normal_trining_16_res_attnmap

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 51989 lora_mytraining.py \
  --process_title parksooyeon --max_token_length 225 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train/bad \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test/bad --seed 42 --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --logging_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/8_lora_diff_trining_16_res_attnmap/logs \
  --wandb_run_name 8_lora_diff_trining_16_res_attnmap --cross_map_res [16] \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/8_lora_diff_trining_16_res_attnmap