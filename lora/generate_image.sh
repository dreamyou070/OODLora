#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50439 generate_image.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5 \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --prompt 'contamination' --sample_sampler ddim \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/bagel --resolution 512,512 --seed 42 \
  --network_weights ../result/MVTec3D-AD_experiment/bagel/lora_training/7_anormal_sample_training_not_use_attn_loss_more_hole/models --num_ddim_steps 50



