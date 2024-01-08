#!/bin/bash
# dreamyou070
# qkrtndus0701?!
# srun -p suma_a6000 -q big_qos --gres=gpu:1 --time=48:00:00 --pty bash -i
# srun -p suma_a6000 -q big_qos --gres=gpu:2 --time=48:00:00 --pty bash -i
# srun -p suma_rtx4090 -q big_qos --gres=gpu:1 --time=48:00:00 --pty bash -i
# srun -p suma_rtx4090 -q big_qos --gres=gpu:2 --time=48:00:00 --pty bash -i
# cd ./Lora/OODLora/lora/
# conda activate venv_lora

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 51234 inference.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 \
  --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --resolution '512,512' \
  --seed 42 \
  --num_ddim_steps 50 \
  --cross_map_res [64] \
  --trg_position "['up']" \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/potato \
  --network_weights ../result/MVTec3D-AD_experiment/potato/lora_training/res_64_up_normal_10_anormal_80/models


# -------------------------------------------------------------------------------------------------------------------------------------------- #
NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 51661 inference_cookie.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 \
  --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --resolution '512,512' \
  --seed 42 \
  --num_ddim_steps 50 \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/cookie \
  --network_weights ../result/MVTec3D-AD_experiment/cookie/lora_training/res_64_up_16_up_down_up_normal_10_anormal_80/models