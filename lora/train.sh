#!/bin/bash
# dreamyou070
# qkrtndus0701?!
# conda activate venv_lora
# cd ./Lora/OODLora/lora/
# srun -p suma_a6000 -q big_qos --job-name=lora_10_2 --gres=gpu:4 --pty bash -i

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50189 train.py \
  --process_title parksooyeon --max_token_length 225 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 1 --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train_ex/bad \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test_ex/bad --seed 42 --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --logging_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/1_lora_normal_trining_all_res_attnmap_attn_loss/logs --normal_training \
  --wandb_run_name 1_lora_normal_trining_all_res_attnmap_attn_loss --cross_map_res [64,32,16,8] \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/1_lora_normal_trining_all_res_attnmap_attn_loss