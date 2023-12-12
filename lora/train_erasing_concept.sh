accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_config/gpu_6_7_config --main_process_port 26724 train_erasing_concept2.py \
  --logging_dir ../result/logs --process_title parksooyeon --max_token_length 225 \
  --log_with wandb --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --wandb_init_name 'bagel_normal_concept_trainingiom_pretraining' --wandb_run_name 1_contrastive_learning_eps_0.00005 \
  --seed 42 --output_dir ../result/MVTec_experiment/bagel/1_contrastive_learning_eps_0.00005 \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --network_weights '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/5_strong_good_training_high_repeat/models/last.safetensors' \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/MyData/object/bagel_inference.txt \
  --max_train_steps 48000 --train_data_dir r'/data7/sooyeon/MyData/anomaly_detection/MVTecAD/paired_data/bad' --resolution 512,512  \
  --train_data_dir /data7/sooyeon/MyData/anomaly_detection/MVTecAD/bad \
  --trg_concept 'good' --sample_prompts /data7/sooyeon/MyData/object/bagel_inference.txt --contrastive_eps 0.00005
# ---------------------------------------------------------------------------------------------------------------------------------------
accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_config/gpu_4_5_config --main_process_port 24524 train_erasing_concept2.py \
  --logging_dir ../result/logs --process_title parksooyeon --max_token_length 225 \
  --log_with wandb --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
  --wandb_init_name 'bagel_normal_concept_trainingiom_pretraining' --wandb_run_name 2_contrastive_learning_eps_0.0005 \
  --seed 42 --output_dir ../result/MVTec_experiment/bagel/2_contrastive_learning_eps_0.0005 \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --train_batch_size 2 \
  --network_weights '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/5_strong_good_training_high_repeat/models/last.safetensors' \
  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 \
  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/MyData/object/bagel_inference.txt \
  --max_train_steps 48000 --train_data_dir r'/data7/sooyeon/MyData/anomaly_detection/MVTecAD/paired_data/bad' --resolution 512,512  \
  --train_data_dir /data7/sooyeon/MyData/anomaly_detection/MVTecAD/bad \
  --trg_concept 'good' --sample_prompts /data7/sooyeon/MyData/object/bagel_inference.txt --contrastive_eps 0.0005