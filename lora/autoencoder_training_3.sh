#!/bin/bash
# start of the script
#!/bin/bash
# sbatch -q big_qos --nodes 2 --output=../result/printing_screen/2_contrastive_learning_eps_0.0_new_code_highrepeat_test.txt training_contrastive.sh
# sbatch -q big_qos --nodes 2 training_contrastive.sh
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

source ~/.bashrc
conda activate venv
ml purge
ml load cuda/11.0

accelerate launch --config_file ../../../gpu_config/gpu_0_1_1_config --main_process_port 50189 autoencoder_training_3.py \
  --logging_dir ../result/logs --process_title parksooyeon --max_token_length 225 \
  --log_with wandb --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --wandb_init_name bagel_training --wandb_run_name 2_vae_decoder_training_uniform_initialize \
  --seed 42 --output_dir ../result/MVTec_experiment/bagel/2_vae_decoder_training_uniform_initialize \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --resolution 512,512 \
  --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --train_batch_size 2 \
  --max_train_steps 100000 \
  --train_data_dir ../../../MyData/anomaly_detection/autoencoder_training/bad \
  --class_caption 'good' \
  --contrastive_eps 0.0 \
  --perceptual_weight 0.001