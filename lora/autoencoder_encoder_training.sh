#!/bin/bash
# start of the script
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

# dreamyou070
# qkrtndus0701?!
# srun -p suma_a6000 -q big_qos --gres=gpu:4 --job-name=cable_4 --pty bash -i
# conda activate venv_lora
# cd ./Lora/OODLora/lora

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_config --main_process_port 50289 autoencoder_encoder_training.py \
  --process_title parksooyeon --max_token_length 225 --log_with wandb --seed 42 \
  --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --resolution 512,512 --save_every_n_epochs 1 --sample_every_n_epochs 1 --train_batch_size 2 --max_train_steps 100000 --start_epoch 0 \
  --wandb_init_name cable_gland_training --log_with wandb \
  --wandb_run_name 2_TS_encoder_normal_anormal_aug \
  --logging_dir ../result/MVTec3D-AD_experiment/cable_gland/vae_training/2_TS_encoder_normal_anormal_aug \
  --output_dir ../result/MVTec3D-AD_experiment/cable_gland/vae_training/2_TS_encoder_normal_anormal_aug \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL/cable_gland/train/bad \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL/cable_gland/test/bad
