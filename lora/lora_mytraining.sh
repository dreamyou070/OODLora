#!/bin/bash

# sbatch -q big_qos --nodes 2 --output=../result/printing_screen/2_contrastive_learning_eps_0.0_new_code_highrepeat_test.txt training_contrastive.sh
# sbatch -p big_suma_rtx3090 --qos big_qos --ntasks-per-node 1 --output /home/dreamyou070/Lora/OODLora/result/printing_screen/test_log.log   --cpus-per-gpu=6 train_contrastive_2.sh
# 1) gpu config : nprocess 2
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 51189 lora_mytraining.py \
  --process_title parksooyeon --max_token_length 225 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train/bad \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test/bad --seed 42 \
  --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --logging_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/1_lora_normal_trining_all_res_attnmap/logs --normal_training \
  --wandb_run_name 1_lora_normal_trining_all_res_attnmap --cross_map_res [64,32,16,8] \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/1_lora_normal_trining_all_res_attnmap

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 51689 lora_mytraining.py \
  --process_title parksooyeon --max_token_length 225 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train/bad \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test/bad --seed 42 \
  --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --logging_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/6_lora_diff_trining_all_res_attnmap/logs \
  --wandb_run_name 6_lora_diff_trining_all_res_attnmap --cross_map_res [64,32,16,8] \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/6_lora_diff_trining_all_res_attnmap

























NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 50289 lora_mytraining.py \
  --logging_dir ../result/logs --process_title parksooyeon --max_token_length 225 \
  --log_with wandb --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --wandb_init_name potato_training \
  --wandb_run_name 2_2_lora_trining_using_noise_diff_map_res_16 \
  --seed 42 \
  --output_dir ../result/MVTec3D-AD_experiment/potato/unet_training/2_2_lora_trining_using_noise_diff_map_res_16 \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution 512,512 \
  --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --sample_prompts ../../../MyData/object/inference.txt \
  --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL/potato/train/bad \
  --class_caption 'good' \
  --contrastive_eps 0.0 \
  --start_epoch 0 \
  --cross_map_res 16

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 53289 lora_mytraining.py \
  --logging_dir ../result/logs --process_title parksooyeon --max_token_length 225 --seed 42 \
  --log_with wandb --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --wandb_init_name cable_gland_training \
  --wandb_run_name 2_lora_trining_normal_data_detail_objective \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution 512,512 --save_every_n_epochs 1 --sample_every_n_epochs 1 \
  --sample_prompts ../../../MyData/object/inference.txt --max_train_steps 48000 \
  --output_dir ../result/MVTec3D-AD_experiment/cable_gland/unet_training/2_lora_trining_normal_data_detail_objective \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL/cable_gland/train_normal/bad \
  --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --normal_training