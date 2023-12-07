python image_inverting.py --device cuda:1 \
  --process_title parksooyeon \
  --output_dir '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_first_experiment' \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 \
  --network_weights '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_first_experiment/last.safetensors' \
  --prompt 'normal' --concept_image_folder /data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb \
  --inversion_experiment --sample_sampler ddim --num_ddim_steps 50 --threshold_time 0







  --negative_prompt 'low quality, worst quality, bad anatomy,bad composition, poor, low effort' \
  --seed 42 --guidance_scale 8 --num_ddim_steps 50 --min_value 12 --max_self_input_time 20 --self_key_control --sample_sampler ddpm