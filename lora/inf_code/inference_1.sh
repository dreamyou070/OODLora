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

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 56389 inference_1.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --prompt 'good' --sample_sampler ddim \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL/bagel \
  --student_pretrained_dir /home/dreamyou070/Lora/OODLora/result/MVTec3D-AD_experiment/potato/vae_training/4_TS_encoder_normal_anormal_no_aug/vae_student_model/student_epoch_000018.pth \
  --network_weights ../result/MVTec3D-AD_experiment/potato/unet_training/1_lora_trining_using_noise_diff/models \
  --output_dir ../result/MVTec3D-AD_experiment/potato/unet_training/1_lora_trining_using_noise_diff/unet_training_lora_finding \
  --resolution 512,512 \
  --seed 42 \
  --mask_thredhold 0 \
  --num_ddim_steps 50 \
  --unet_only_inference_times 25