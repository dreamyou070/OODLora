#!/bin/bash

# sbatch -q big_qos --nodes 2 --output=../result/printing_screen/2_contrastive_learning_eps_0.0_new_code_highrepeat_test.txt training_contrastive.sh
# sbatch -q big_qos --nodes 2 training_contrastive.sh

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0
# srun -p suma_rtx4090 -q big_qos --job-name=s_learn_recon --gres=gpu:2 --pty bash -i

accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 57689 autoencoder_contrastive_learning.py \
  --process_title parksooyeon --max_token_length 225 \
  --logging_dir ../result/logs \
  --log_with wandb \
  --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --wandb_init_name bagel_training \
  --wandb_run_name 3_TS_test_contrastive_recon_loss \
  --seed 42 \
  --output_dir ../result/MVTec_experiment/bagel/vae_training/3_TS_test_contrastive_recon_loss \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --resolution 512,512 \
  --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --train_batch_size 3 \
  --max_train_steps 100000 \
  --student_reconst_loss \
  --train_data_dir ../../../MyData/anomaly_detection/VisA/MVTecAD/paired_data/bad