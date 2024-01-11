#!bin/bash
# dreamyou070
# qkrtndus0701?!
# srun -p suma_a6000 -q big_qos --gres=gpu:2 --time=2-0 --pty bash -i
# srun -p suma_a6000 -q big_qos --gres=gpu:2 --time=1-0 --pty bash -i
# srun -p suma_rtx4090 -q big_qos --gres=gpu:2 --time=1-0 --pty bash -i
# cd ./Lora/OODLora/lora/
# conda activate venv_lora

class_name="carrot"
data_source='train_ex'
save_folder="1_7_res_64_down_32_up_from_normal_eighth_epoch_anormal_deact_only_on_64_down"
start_folder="0_7_res_64_down_32_up_normal"
trg_lora_model="epoch-000008.safetensors"
start_epoch=0
port_number=55888

train_data_dir="../../../MyData/anomaly_detection/MVTec3D-AD/${class_name}/${data_source}/rgb"
output_dir="../result/MVTec3D-AD_experiment/${class_name}/lora_training/${save_folder}"
start_dir="../result/MVTec3D-AD_experiment/${class_name}/lora_training/${start_folder}"
network_weights="${start_dir}/models/${trg_lora_model}"


NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port $port_number train.py \
  --process_title parksooyeon \
  --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc  \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 --train_batch_size 1 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 --resolution '512,512' --save_every_n_epochs 1 \
  --sample_every_n_epochs 1 \
  --sample_prompts ../../../MyData/anomaly_detection/inference.txt \
  --max_train_steps 300000 --use_attn_loss --task_loss_weight 1.0 --seed 42 --class_caption 'good' --start_epoch 0 \
  --wandb_init_name "$class_name" \
  --train_data_dir "$train_data_dir" \
  --start_epoch $start_epoch \
  --output_dir "$output_dir" \
  --cross_map_res [64]\
  --trg_position "['down']" \
  --trg_part "['attn_0','attn_1','attn_2']" \
  --network_weights "$network_weights"