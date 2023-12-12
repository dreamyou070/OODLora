python image_inverting.py --device cuda:2 \
  --process_title parksooyeon \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --network_weights '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_contrastive_learning_eps_0.00005/epoch-000065.safetensors' \
  --prompt 'good' --inversion_experiment --sample_sampler ddim --num_ddim_steps 50 \
  --output_dir '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_contrastive_learning_eps_0.00005/inference_check' \
  --concept_image_folder /data7/sooyeon/MyData/anomaly_detection/MVTecAD/bagel \
  --latent_coupling --repeat_time 51 --threshold_time 1000
# ---------------------------------------------------------------------------------------------------------------------------------------
python image_inverting_new_scheduler.py --device cuda:2 \
  --process_title parksooyeon \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora \
  --network_dim 64 --network_alpha 4 \
  --network_weights '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_contrastive_learning_eps_0.00005/epoch-000065.safetensors' \
  --prompt 'good' --inversion_experiment --sample_sampler ddim --num_ddim_steps 50 \
  --output_dir '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/1_contrastive_learning_eps_0.00005/inference_check' \
  --concept_image_folder /data7/sooyeon/MyData/anomaly_detection/MVTecAD/bagel \
  --latent_coupling --repeat_time 51 --threshold_time 1000
