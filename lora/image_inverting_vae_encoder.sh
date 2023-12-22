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


NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50289 image_inverting_vae_encoder.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --num_ddim_steps 50 \
  --concept_image_folder ../../../MyData/anomaly_detection/VisA/MVTecAD/bagel \
  --output_dir ../result/MVTec_experiment/bagel/vae_training/5_TS_encoder_contrastive_recon_loss/inference/with_lora \
  --student_pretrained_dir ../result/MVTec_experiment/bagel/vae_training/5_TS_encoder_contrastive_recon_loss/vae_student_model/student_epoch_000008.pth \
  --network_weights ../result/MVTec_experiment/bagel/5_lora_trining_using_noise_diff/epoch-000006.safetensors \
  --repeat_time 51 \
  --resolution 512,512 \
  --seed 42 \
  --use_binary_mask \
  --mask_thredhold 0.5 \
  --final_time 100