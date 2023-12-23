#!/bin/bash

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50189 autoencoder_inference_encoder.py \
  --process_title parksooyeon \
  --seed 42 \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --student_encoder_pretrained_dir ../result/MVTec3D-AD_experiment/cable_gland/vae_training/2_TS_enc_dec_contrastive_recon_loss/vae_encoder_student_model/encoder_student_epoch_000005.pth \
  --output_dir ../result/MVTec3D-AD_experiment/cable_gland/vae_training/2_TS_enc_dec_contrastive_recon_loss/only_encoder \
  --anormal_folder ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL/cable_gland \
  --resolution 512,512 \
  --training_data_check
