accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_config/gpu_6_7_config --main_process_port 26764 check_lora_size.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --max_token_length 225 \
                  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
                  --output_dir ../result/perfusion_experiment/cat/iom_pretrain_sen_10_unet_inlayers_heatmap_backprop_infer_inlayers_weight_diff_loss_10 \
                  --save_folder_name 'test' --network_module networks.lora --efficient_layer 'text,unet' \
                  --org_weight_alpha 0.0 --lora_weight_alpha 0.5 --total_weight_alpha 0.0 --histogram_save_folder_name histogram_unet_lora_up_down_total_self_condition_small_scale_total_up  \
                  --lora_down_weight_alpha 0.5 --lora_up_weight_alpha 0.5



