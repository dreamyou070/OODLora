#!/bin/bash
# dreamyou070
# qkrtndus0701?!
# cd ./Lora/OODLora/lora/
# srun -p suma_a6000 -q big_qos --job-name=lora_train_5 --gres=gpu:2 --time=48:00:00 --pty bash -i
# conda activate venv_lora


NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 52689 train.py \
  --process_title parksooyeon --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 1 --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train_ex/bad --task_loss_weight 1.0 \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test_ex/bad --seed 42 --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/13_normal_sample_training --cross_map_res [64,32,16,8] --use_attn_loss

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 51489 train_14.py \
  --process_title parksooyeon --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 1 --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train_ex_14/bad --task_loss_weight 1.0 \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test_ex/bad --seed 42 --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/14_normal_sample_training_score_check \
  --cross_map_res [64,32,16,8] --use_attn_loss --normal_activation_train

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 54321 train_15.py \
  --process_title parksooyeon \
  --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name cable_gland_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 1 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 \
  --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/carrot/train_ex/rgb --task_loss_weight 1.0 \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/carrot/test_ex/rgb \
  --seed 42 \
  --class_caption 'good' \
  --start_epoch 0 \
  --output_dir ../result/MVTec3D-AD_experiment/carrot/lora_training/res_64_up \
  --cross_map_res [64] --use_attn_loss --normal_activation_train --trg_position "['up']"














NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 51212 train_layerwise_text_embedding.py \
  --process_title parksooyeon --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora_emgedding \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 1 --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train_ex_15/bad --task_loss_weight 1.0 \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test_ex/bad --seed 42 --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/res_64_32_up_text_embedding --network_train_unet_only \
  --cross_map_res [64,32] --use_attn_loss --normal_activation_train --trg_position "['up']"