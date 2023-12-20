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


accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 50189 training_end_to_end.py \
  --logging_dir ../result/logs --process_title parksooyeon --max_token_length 225 \
  --log_with wandb --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --wandb_init_name bagel_training --wandb_run_name 4_train_end_to_end \
  --seed 42 --output_dir ../result/MVTec_experiment/bagel/4_train_end_to_end \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 1 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution 512,512 \
  --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --sample_prompts ../../../MyData/object/bagel_inference.txt \
  --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/VisA/MVTecAD/paired_data/bad \
  --class_caption 'good' \
  --contrastive_eps 0.0