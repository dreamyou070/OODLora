NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 52121 ad_recon.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --resolution '512,512' \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/dowel \
  --seed 42 \
  --network_weights ../result/MVTec3D-AD_experiment/dowel/lora_training/2_res_64_up_16_up_normal_10_anormal_50/models \
  --num_ddim_steps 50 \
  --trg_lora_epoch 'epoch-000017.safetensors' \
  --inner_iter 10 --only_zero_save \
  --cross_map_res [64] --trg_position "['up']" --trg_part "attn_0"