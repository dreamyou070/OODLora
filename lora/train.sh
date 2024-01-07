#!/bin/bash
# dreamyou070
# qkrtndus0701?!
# srun -p suma_rtx4090 -q big_qos --job-name=lora_train_5 --gres=gpu:1 --time=48:00:00 --pty bash -i
# cd ./Lora/OODLora/lora/
# conda activate venv_lora

# 7) 32,16res up
# 8) 32,16res down
# 9) 32,16res updown

# 64up, 32res, up
# 64up, 32res, down
# 64up, 32res, updown

# 64up, 16res, up
# 64up, 16res, down
# 64up, 16res, updown

# 64up, 32,16res up
# 64up, 32,16res down
# 64up, 32,16res updown


NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 50006 train.py \
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
  --use_attn_loss \
  --task_loss_weight 1.0 \
  --seed 42 \
  --class_caption 'good' \
  --start_epoch 0 \
  --wandb_init_name cookie \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/cookie/train_ex/rgb \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/cookie/test_ex/rgb \
  --output_dir ../result/MVTec3D-AD_experiment/cookie/lora_training/res_16_up_down --cross_map_res [16]  --trg_position "['up','down']"