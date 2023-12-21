#!/bin/bash
#SBATCH --partition=compute-od-gpu
#SBATCH --job-name=intelmpi_test
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --comment "Key=Monitoring,Value=ON"
#SBATCH --exclusive

# sbatch -q big_qos --nodes 2 --output=../result/printing_screen/2_contrastive_learning_eps_0.0_new_code_highrepeat_test.txt training_contrastive.sh
# sbatch -p big_suma_rtx3090 --qos big_qos --ntasks-per-node 1 --output /home/dreamyou070/Lora/OODLora/result/printing_screen/test_log.log   --cpus-per-gpu=6 train_contrastive_2.sh


# 1) gpu config : nprocess 2

echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
source ~/.bashrc
conda activate venv_lora
ml purge
ml load cuda/11.0


accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50189 autoencoder_inference_encoder.py \
  --process_title parksooyeon \
  --seed 42 \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --student_pretrained_dir ../result/MVTec_experiment/bagel/vae_training/4_TS_encoder_test_contrastive_recon_loss/vae_student_model/student_epoch_000003.pth \
  --output_dir ../result/MVTec_experiment/bagel/vae_training/4_TS_encoder_test_contrastive_recon_loss \
  --anormal_folder ../../../MyData/anomaly_detection/VisA/MVTecAD/bagel/test \
  --resolution 512,512