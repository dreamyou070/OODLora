#!/bin/bash
# dreamyou070
# qkrtndus0701?!
# cd ./Lora/OODLora/lora/
# srun -p suma_a6000 -q big_qos --job-name=lora_train_5 --gres=gpu:2 --time=48:00:00 --pty bash -i
# conda activate venv_lora

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 50545 train.py \
  --process_title parksooyeon \
  --log_with wandb \
  --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 --train_batch_size 1 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' \
  --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --sample_prompts ../../../MyData/anomaly_detection/inference.txt \
  --max_train_steps 100000 \
  --cross_map_res [64]  --trg_position "['up']" \
  --use_attn_loss \
  --task_loss_weight 1.0 \
  --seed 42 \
  --class_caption 'good' \
  --start_epoch 0 \
  --wandb_init_name dowel \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/dowel/train_ex/rgb \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/dowel/test_ex/rgb \
  --output_dir ../result/MVTec3D-AD_experiment/dowel/lora_training/res_64_up


  --network_weights ../result/MVTec3D-AD_experiment/foam/lora_training/res_64_up/models/epoch-000004.safetensors \