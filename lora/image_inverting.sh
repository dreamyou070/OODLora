python image_inverting.py --device cuda:1 \
  --process_title parksooyeon \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --network_weights '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/3_strong_good_training/last.safetensors' \
  --prompt 'good' --inversion_experiment --sample_sampler ddim --num_ddim_steps 50 \
  --output_dir '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/3_strong_good_training/inference_check' \
  --concept_image_folder /data7/sooyeon/MyData/anomaly_detection/MVTecAD/bagel \
  --repeat_time 50 --threshold_time 1000 --cfg_check 1000 --inversion_weight 3.0