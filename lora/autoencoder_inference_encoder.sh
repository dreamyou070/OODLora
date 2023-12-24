#!/bin/bash

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50189 autoencoder_inference_encoder.py \
  --process_title parksooyeon \
  --seed 42 \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --student_encoder_pretrained_dir ../result/MVTec3D-AD_experiment/potato/vae_training/4_TS_encoder_patchwise_augmenting/vae_student_model/student_epoch_000018.pth \
  --output_dir ../result/MVTec3D-AD_experiment/potato/vae_training/4_TS_encoder_patchwise_augmenting/only_encoder_decoder_pretrained \
  --anormal_folder ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL/potato \
  --resolution 512,512 \
  --training_data_check