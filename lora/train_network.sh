accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_config/gpu_4_5_config --main_process_port 24524 train_network_sy.py \
                  --logging_dir ../result/logs --process_title parksooyeon --max_token_length 225 \
                  --seed 42 --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
                  --output_dir ../result/perfusion_experiment/cat/iom_pretrain_10_unet_inlayers_heatmap_backprop_infer_key_value_reg_high_preservating_loss_ratio \
                  --train_data_dir /data7/sooyeon/MyData/perfusion_dataset/cat/iom_1 \
                  --mask_dir /data7/sooyeon/MyData/perfusion_dataset/cat/iom_1_mask \
                  --log_with wandb \
                  --wandb_init_name iom_pretraining --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
                  --wandb_run_name iom_pretrain_10_unet_inlayers_heatmap_backprop_infer_key_value_reg_high_preservating_loss_ratio \
                  --class_token 'cat' --class_caption 'cat' --trg_concept iom --class_caption_dir /data7/sooyeon/MyData/perfusion_dataset/cat/cat_sentences.txt \
                  --network_module networks.lora --resolution 512,512 --net_key_names text --network_dim 64 --network_alpha 4 --train_batch_size 1 \
                  --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
                  --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 --pretraining_epochs 10 --unet_net_key_names 'unet' \
                  --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/MyData/perfusion_dataset/cat/iom_inference.txt \
                  --heatmap_loss --mask_threshold 0.5 --first_second_training --attn_loss_ratio 1 --heatmap_backprop \
                  --efficient_layer 'text,down_blocks_2,mid,up_blocks_1' --save_folder_name 'inlayer' --class_preserving --class_preserving_ratio 10.0 \
                  --class_lora_preserving --class_lora_preserving_ratio 10.0

















accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_config/gpu_6_7_config --main_process_port 26764 train_network_pretraining.py \
    --logging_dir ../result/logs --process_title parksooyeon \
    --seed 42 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
    --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5 \
    --wandb_init_name iom_pretraining \
    --wandb_run_name iom_pretrain_sen_10_inlayers_heatmap_backprop_infer_inlayers_without_attn_loss \
    --output_dir ../result/perfusion_experiment/cat/iom_pretrain_sen_10_inlayers_heatmap_backprop_infer_inlayers_without_attn_loss \
    --train_data_dir /data7/sooyeon/MyData/perfusion_dataset/cat/iom_1 \
    --mask_dir /data7/sooyeon/MyData/perfusion_dataset/cat/iom_1_mask \
    --class_token 'cat' --class_caption 'cat' --trg_concept iom --class_caption_dir /data7/sooyeon/MyData/perfusion_dataset/cat/cat_sentences.txt \
    --network_module networks.lora --resolution 512,512 --net_key_names text --network_dim 64 --network_alpha 4 --train_batch_size 2 \
    --optimizer_type AdamW --lr_scheduler cosine_with_restarts --lr_warmup_steps 144 \
    --learning_rate 0.0003 --unet_lr 0.0001 --text_encoder_lr 0.00005 --pretraining_epochs 10 --unet_net_key_names 'down_blocks_2,mid,up_blocks_1' \
    --save_every_n_epochs 1 --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/MyData/perfusion_dataset/cat/iom_inference.txt \
    --heatmap_loss --mask_threshold 0.5 --first_second_training --attn_loss_ratio 10 \
    --efficient_layer 'text,down_blocks_2,mid,up_blocks_1' --save_folder_name 'inlayer'










# inlayer, condition 만을 가지고 inference 한 경우인데 잘 안나온다.


#
accelerate launch --config_file /data7/sooyeon/LyCORIS/gpu_4_5_config --main_process_port 24564 only_inference.py \
                  --logging_dir ./result/logs --process_title parksooyeon \
                  --max_token_length 225 \
                  --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
                  --output_dir ./result/perfusion_experiment/teddy_bear/te_pretrain_581_sen_10_attn1_to_k_attn1_to_v_attn2_to_k_attn2_to_v \
                  --save_folder_name 'efficient_inlayer_condition_proj_in_ff_net' \
                  --network_module networks.lora --efficient_layer 'text,down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k,down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v,down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k,down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v,down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k,down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v,down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k,down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v,mid_block_attentions_0_transformer_blocks_0_attn1_to_k,mid_block_attentions_0_transformer_blocks_0_attn1_to_v,mid_block_attentions_0_transformer_blocks_0_attn2_to_k,mid_block_attentions_0_transformer_blocks_0_attn2_to_v,up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k,up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v,up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k,up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v,up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k,up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v,up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k,up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v,up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_k,up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_v,up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_k,up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_v,proj_in,ff_net' \
                  --sample_every_n_epochs 1 --sample_prompts /data7/sooyeon/LyCORIS/test/test_td.txt