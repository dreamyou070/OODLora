python customizing_scheduling.py --device cuda:5 \
  --process_title parksooyeon \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 --prompt 'good' --sample_sampler ddim --num_ddim_steps 50 \
  --output_dir '/data7/sooyeon/Lora/OODLora/result' \
  --network_weights /data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/model/epoch-000003.safetensors \
  --concept_image_folder /data7/sooyeon/MyData/anomaly_detection/MVTecAD/bagel