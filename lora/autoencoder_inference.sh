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

accelerate launch --config_file ../../../gpu_config/gpu_3_4_config --main_process_port 53489 autoencoder_inference.py \
  --device 'cuda:4' \
  --process_title parksooyeon \
  --seed 42 \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --vae_pretrained_dir ../result/MVTec_experiment/bagel/vae_training/vae_model/vae_epoch_000006/pytorch_model.bin \
  --output_dir ../result/MVTec_experiment/bagel/vae_training \
  --resolution 512,512