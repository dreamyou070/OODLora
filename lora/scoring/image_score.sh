python image_score.py --device cuda:0 \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --network_weights '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_first_experiment/last.safetensors' \
  --output_dir '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_first_experiment/scoring_test'