NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 52121 ad_recon.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --prompt 'good' --sample_sampler ddim \
  --resolution '512,512' \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/bagel \
  --seed 42 \
  --network_weights ../result/MVTec3D-AD_experiment/bagel/lora_training/res_64_up/models \
  --num_ddim_steps 50 \
  --cross_map_res [64] \
  --pixel_copy

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 52121 ad_recon_inner_iter.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --resolution '512,512' \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/bagel \
  --seed 42 \
  --network_weights ../result/MVTec3D-AD_experiment/bagel/lora_training/res_64_up/models \
  --num_ddim_steps 50 \
  --cross_map_res [64] \
  --pixel_copy \
  --inner_iter 20