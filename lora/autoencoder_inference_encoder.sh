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

accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50189 autoencoder_inference.py \
  --process_title parksooyeon \
  --seed 42 \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --student_pretrained_dir ../result/MVTec_experiment/bagel/vae_training/4_TS_encoder_test_contrastive_recon_loss/vae_student_model/student_epoch_000003.pth \
  --output_dir ../result/MVTec_experiment/bagel/vae_training/4_TS_encoder_test_contrastive_recon_loss \
  --anormal_folder ../../../MyData/anomaly_detection/VisA/MVTecAD/bagel/test \
  --resolution 512,512