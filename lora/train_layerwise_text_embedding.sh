NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_1_config --main_process_port 51663 train_layerwise_text_embedding.py \
  --process_title parksooyeon --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc --wandb_init_name bagel_lora_emgedding \
  --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 1 --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --resolution '512,512' --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts ../../../MyData/anomaly_detection/inference.txt --max_train_steps 48000 \
  --train_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/train_ex_15/bad --task_loss_weight 1.0 \
  --valid_data_dir ../../../MyData/anomaly_detection/MVTec3D-AD/bagel/test_ex/bad --seed 42 --class_caption 'good' --contrastive_eps 0.0 --start_epoch 0 \
  --output_dir ../result/MVTec3D-AD_experiment/bagel/lora_training/res_64_32_16_up_down_text_embedding_without_64_down --network_train_unet_only \
  --cross_map_res [64,32,16] --use_attn_loss --normal_activation_train --trg_position "['up','down']" --detail_64