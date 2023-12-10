cuda_visible_devices=6 python gen_img_diffusers.py \
  --network_module networks.lora \
  --ckpt '/data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors' \
  --network_weights "/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/5_strong_good_training_high_repeat/models/last.safetensors" \
  --outdir "/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/5_strong_good_training_high_repeat/generating" \
  --seed 100 --prompt "good" --H 512 --W 512 --clip_skip 2 --scale 10