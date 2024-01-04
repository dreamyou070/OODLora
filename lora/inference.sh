#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 53889 inference.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --prompt 'good' --sample_sampler ddim \
  --resolution '512,512' --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/bagel --resolution 512,512 --seed 42 \
  --network_weights ../result/MVTec3D-AD_experiment/bagel/lora_training/17_normal_sample_training_res_64_change_to_max/models \
  --num_ddim_steps 50 --final_noising_time 980 --pixel_mask_res 32 --pixel_thred 0.3 --cross_map_res [64,32,16,8] --org_latent_attn_map_check