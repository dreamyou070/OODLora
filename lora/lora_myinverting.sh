#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 51189 lora_myinverting.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/bagel \
  --student_pretrained_dir ../result/MVTec3D-AD_experiment/bagel/vae_training/4_TS_encoder_normal_no_aug/vae_student_model/student_epoch_000150.pth \
  --network_weights ../result/MVTec3D-AD_experiment/bagel/lora_training/8_lora_diff_trining_32_res_attnmap/epoch-000001.safetensors \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/inference \
  --resolution 512,512 \
  --seed 42 \
  --cross_map_res [32] \
  --mask_thredhold 0.01 \
  --num_ddim_steps 50 \
  --unet_only_inference_times 50