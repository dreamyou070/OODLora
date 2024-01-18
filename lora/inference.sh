#!/bin/bash

# dreamyou070
# qkrtndus0701?!
# srun -p suma_a6000 -q big_qos --gres=gpu:1 --time=2-0 --pty bash -i
# srun -p suma_a6000 -q big_qos --gres=gpu:2 --time=2-0 --pty bash -i
# srun -p suma_rtx4090 -q big_qos --gres=gpu:2 --time=2-0 --pty bash -i
# cd ./Lora/OODLora/lora/
# conda activate venv_lora

class_name='bagel'
start_epoch=0
port_number=53113
folder_name="2_2_res_8"
concept_image_folder="../../../MyData/anomaly_detection/MVTec3D-AD/${class_name}"
output_dir="../result/MVTec3D-AD_experiment/${class_name}/lora_training/anormal/${folder_name}"
network_weights="${output_dir}/models"

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port ${port_number} inference.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --prompt 'good' --sample_sampler ddim --resolution '512,512' --seed 42 \
  --cross_map_res [8] \
  --trg_position "['mid']" \
  --concept_image_folder "${concept_image_folder}" \
  --network_weights "${network_weights}" \
  --truncate_length 2 \
  --start_epoch ${start_epoch}
