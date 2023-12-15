#!/bin/bash
# start of the script

#SBATCH --job-name=parksooyeon_job
#SBATCH --gres=gpu:4
#SBTACH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=example.out
#SBATCH --time 0-23:00:00
launch --config_file /data7/sooyeon/LyCORIS/gpu_config/gpu_4_5_config
accelerate config ~~~ training_contrastive.py \
  --logging_dir ../result/logs --process_title parksooyeon --max_token_length 225 \
  --log_with wandb --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --wandb_init_name bagel_training --wandb_run_name 2_contrastive_learning_eps_0.0_new_code_highrepeat \
  --seed 42 --output_dir ../result/MVTec_experiment/bagel/2_contrastive_learning_eps_0.0_new_code_highrepeat \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution 512,512 \
  --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --sample_prompts ../../../MyData/object/bagel_inference.txt \
  --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTecAD/paired_data/bad \
  --class_caption 'good' \
  --contrastive_eps 0.0
