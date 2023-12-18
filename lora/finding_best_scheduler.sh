#!/bin/bash

#SBATCH --job-name=parksooyeon_finding_best_scheduler
#SBATCH --gres=gpu:1
#SBATCH --output=../result/printing_screen/noising_scheduler_log.log

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

source ~/.bashrc
conda activate venv
ml purge
ml load cuda/11.0
# sbatch -p dell_rtx3090 -q big_qos --nodes 1 --output=../result/printing_screen/noising_scheduler_log.log finding_best_scheduler.sh
python finding_best_scheduler.py --device cuda \
   --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
   --network_module networks.lora \
   --network_dim 64 --network_alpha 4 \
   --prompt 'good' \
   --sample_sampler ddim --num_ddim_steps 50 --output_dir ../../../result \
   --network_weights ../result/MVTec_experiment/bagel/unet_training/model/epoch-000003.safetensors \
   --output_dir ../result/new_alphas_cumprod_noising \
   --concept_image_folder ../../../MyData/anomaly_detection/VisA/MVTecAD/bagel