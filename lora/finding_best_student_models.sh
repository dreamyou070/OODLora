#!/bin/bash

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0


NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 59389 finding_best_student_model.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --student_encoder_pretrained_dir  ../result/MVTec3D-AD_experiment/cable_gland/vae_training/1_TS_encoder_normal_abnormal_aug/vae_student_model \
  --output_dir ../result/MVTec3D-AD_experiment/cable_gland/vae_training/1_TS_encoder_normal_abnormal_aug/inference_finding_best_epoch \
  --anormal_folder ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL/cable_gland \
  --resolution 512,512 \
  --seed 42