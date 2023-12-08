python using_correcting_coefficient.py --device cuda:1 \
  --process_title parksooyeon \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --network_weights '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_first_experiment/last.safetensors' \
  --prompt 'normal' --inversion_experiment --sample_sampler ddim --num_ddim_steps 50 \
  --output_dir '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_first_experiment/train_experience/1_experiment' \
  --concept_image_folder /data7/sooyeon/MyData/anomaly_detection/MVTecAD/bagel/test/contamination/rgb \
  --repeat_time 30 --threshold_time 0