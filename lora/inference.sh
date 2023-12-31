#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50939 inference.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --prompt 'good bad' --sample_sampler ddim \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/bagel --resolution 512,512 --seed 42 \
  --network_weights ../result/MVTec3D-AD_experiment/bagel/lora_training/2_training_16res_with_CrossEntropy/epoch-000002.safetensors \
  --num_ddim_steps 50 --final_noising_time 980 --pixel_mask_res 16 --pixel_thred 0.9 \
  --cross_map_res [16] --other_token_preserving --inner_iteration 1 --mask_thredhold 0.8