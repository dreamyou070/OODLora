accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_config/gpu_6_7_config --main_process_port 26724 train_erasing_concept.py \
  --logging_dir ../result/logs --process_title parksooyeon --max_token_length 225 \
  --log_with wandb --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --wandb_run_name bagel_normal_concept_trainingiom_pretraining --wandb_run_name 6_img_negative_guidance \
  --seed 42 --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --output_dir ../result/MVTec_experiment/bagel/6_img_negative_guidance \
  --network_module networks.lora \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/MyData/object/bagel_inference.txt \
  --max_train_steps 48000







  --train_data_dir /data7/sooyeon/MyData/object/bagel/good \
  --trg_concept 'good' --resolution 512,512 --net_key_names text --network_dim 64 --network_alpha 4 --train_batch_size 1 \
