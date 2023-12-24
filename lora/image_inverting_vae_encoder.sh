#!/bin/bash

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

# if mask_theredhold is big, means more background
# if mask thredhold is small means more LORA

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 51189 image_inverting_vae_encoder.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL/potato \
  --student_pretrained_dir ../result/MVTec3D-AD_experiment/potato/vae_training/1_TS_encoder_patchwise_augmenting/vae_student_model/student_epoch_000055.pth \
  --network_weights ../result/MVTec3D-AD_experiment/potato/unet_training/1_lora_trining_using_noise_diff/epoch-000001.safetensors \
  --output_dir ../result/MVTec3D-AD_experiment/potato/inference \
  --resolution 512,512 \
  --seed 42 \
  --mask_thredhold 0 \
  --num_ddim_steps 50 \
  --unet_only_inference_times 50