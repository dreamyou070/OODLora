#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0
# dreamyou070
# qkrtndus0701?!
# conda activate venv_lora
# cd ./Lora/OODLora/lora/
# srun -p suma_a6000 -q big_qos --gres=gpu:1 --job-name=inf10 --time=48:00:00 --pty bash -i

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 51239 finding_best_lora.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --prompt 'good' --sample_sampler ddim \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/bagel \
  --network_weights ../result/MVTec3D-AD_experiment/bagel/lora_training/2_training_16res_with_CrossEntropy/epoch-000012.safetensors \
  --resolution 512,512 --seed 42 --cross_map_res [16] --pixel_mask_res 16 --mask_thredhold 0 --num_ddim_steps 1000 --final_noising_time  100 --pixel_thred 0