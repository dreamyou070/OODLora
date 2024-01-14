# dreamyou070
# qkrtndus0701?!
# srun -p suma_a6000 -q big_qos --gres=gpu:1 --time=2-0 --pty bash -i
# srun -p suma_a6000 -q big_qos --gres=gpu:2 --time=2-0 --pty bash -i
# cd ./Lora/OODLora/lora/
# conda activate venv_lora

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config --main_process_port 51201 ad_recon.py \
  --process_title parksooyeon --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --prompt 'good' \
  --sample_sampler ddim \
  --resolution '512,512' \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTec3D-AD/carrot \
  --seed 42 \
  --network_weights ../result/MVTec3D-AD_experiment/carrot/lora_training/anormal/res_64_up_down_32_up_down_text_len_3_more_cut_with_background_loss_recode/models \
  --num_ddim_steps 50 \
  --trg_lora_epoch 'epoch-000012.safetensors' \
  --inner_iter 10 --only_zero_save \
  --cross_map_res [64] --trg_position "['down']" --trg_part "attn_2" --use_avg_mask