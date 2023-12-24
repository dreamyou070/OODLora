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


NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 50189 image_inverting_vae_encoder.py \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --num_ddim_steps 50 \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD_Experiment/potato \
  --output_dir ../result/MVTec3D-AD_experiment/potato/inference \
  --student_pretrained_dir ../result/MVTec3D-AD_experiment/potato/vae_training/1_TS_encoder_patchwise_augmenting/vae_student_model/student_epoch_000020.pth \
  --network_weights ../result/MVTec3D-AD_experiment/potato/unet_training/1_lora_trining_using_noise_diff/epoch-000001.safetensors \
  --repeat_time 51 \
  --resolution 512,512 \
  --seed 42 \
  --mask_thredhold 0.3 \
  --final_time 980
