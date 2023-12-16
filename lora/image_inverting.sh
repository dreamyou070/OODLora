python image_inverting.py --device cuda:5 \
  --process_title parksooyeon \
  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --network_module networks.lora --network_dim 64 --network_alpha 4 \
  --network_weights /data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/2_contrastive_learning_eps_0.0_new_code_highrepeat/epoch-000005.safetensors \
  --prompt 'good' \
  --inversion_experiment \
  --sample_sampler ddim \
  --num_ddim_steps 50 \
  --output_dir /data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/2_contrastive_learning_eps_0.0_new_code_highrepeat \
  --concept_image_folder ../../../MyData/anomaly_detection/MVTecAD/bagel \
  --repeat_time 51 \
  --self_attn_threshold_time 10000 \
  --using_customizing_scheduling