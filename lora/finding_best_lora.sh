#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50139 finding_best_lora.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --prompt 'good' --sample_sampler ddim \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/bagel \
  --network_weights ../result/MVTec3D-AD_experiment/bagel/lora_training/not_good_7_lora_diff_trining_64_res_attnmap_attn_loss/epoch-000001.safetensors \
  --resolution 512,512 --seed 42 \
  --cross_map_res [64] --pixel_mask_res 64 \
  --mask_thredhold 0 \
  --num_ddim_steps 1000 \
  --final_noising_time  100 \
  --pixel_thred 0 --other_token_preserving