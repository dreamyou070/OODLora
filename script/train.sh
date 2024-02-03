#! /bin/bash

class_name="bagel"
data_source='train_normal'
data_folder='MVTec3D-AD'
train_data_dir="../../../MyData/anomaly_detection/${data_folder}/${class_name}/${data_source}/rgb"
normal_folder='normal'
save_folder="res_64_up_task_loss_mahal_dist_attn_loss_0.008_actdeact_perlin_noise_second_code_normal_with_back"
output_dir="../result/${data_folder}_experiment/${class_name}/lora_training/${normal_folder}/${save_folder}"
#network_weights="../result/MVTec3D-AD_experiment/${class_name}/lora_training/${normal_folder}/res_64_down_1_mahal_attn_0.001_act_deact/models/epoch-000002.safetensors"
start_epoch=0
port_number=58860

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config \
  --main_process_port $port_number ../lora/train.py --process_title parksooyeon \
  --wandb_init_name ${class_name} --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc  \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 \
  --seed 42 --train_batch_size 1 \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt \
  --start_epoch $start_epoch --max_train_steps 500 --max_train_epochs 500 \
  --train_data_dir "$train_data_dir" --resolution '512,512' --class_caption 'good' \
  --output_dir "$output_dir" \
  --do_task_loss --task_loss_weight 1.0 \
  --do_attn_loss --attn_loss_weight 0.008 --cls_training --back_training \
  --trg_layer_list "['up_blocks_3_attentions_1_transformer_blocks_0_attn2']" \
  --mahalanobis_loss_weight 1 \
  --act_deact --act_deact_weight 1.0 \
  --normal_with_back
  # --add_random_query
  # --mahalanobis_loss_weight 1 \
  # --text_frozen --network_weights "$network_weights"
  # --do_task_loss --task_loss_weight 1 --act_deact --act_deact_weight 1.0 \