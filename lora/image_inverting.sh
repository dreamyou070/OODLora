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

python image_inverting.py --device cuda \
  --process_title parksooyeon \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 \
  --network_weights ../result/MVTec_experiment/bagel/unet_training/model/epoch-000003.safetensors \
  --prompt 'good' \
  --inversion_experiment \
  --sample_sampler ddim \
  --num_ddim_steps 50 \
  --output_dir ../result/MVTec_experiment/bagel/unet_training/new_alphas_cumprod_noising \
  --concept_image_folder ../../../MyData/anomaly_detection/VisA/MVTecAD/bagel \
  --repeat_time 51 \
  --self_attn_threshold_time 1000 \
  --final_time 100 \
  --with_new_noising_alphas_cumprod