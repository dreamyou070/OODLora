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

accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 53489 autoencoder_inference.py \
  --device 'cuda:4' \
  --process_title parksooyeon \
  --seed 42 \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --student_pretrained_dir ../result/MVTec_experiment/bagel/vae_training/1_TS_test/vae_student_model/student_epoch_000001.pth \
  --output_dir ../result/MVTec_experiment/bagel/vae_training/1_TS_test/vae_student_model \
  --resolution 512,512





