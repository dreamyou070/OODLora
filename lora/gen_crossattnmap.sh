CUDA_VISIBLE_DEVICES=2 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/perfusion_experiment/teddy_bear/te_pretrain_581_sen_10_unet_inlayers_cond_inference_classcaption_preserving_on_inlayer_heatmap_backprop_attn_loss_ratio_10_highrepeat/last.safetensors' \
      --outdir './result/perfusion_experiment/teddy_bear/te_pretrain_581_sen_10_unet_inlayers_cond_inference_classcaption_preserving_on_inlayer_heatmap_backprop_attn_loss_ratio_10_highrepeat/inference_test/last_epoch' \
      --from_file '/data7/sooyeon/LyCORIS/test/td_inference_temp.txt' \
      --efficient_layer 'text,unet' \
      --trg_token 'sitting'

CUDA_VISIBLE_DEVICES=3 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/perfusion_experiment/teddy_bear/te_pretrain_20_unet_down_2_mid_up_1/last.safetensors' \
      --outdir './result/perfusion_experiment/teddy_bear/te_pretrain_20_unet_down_2_mid_up_1/inference/epoch_16/' \
      --from_file '/data7/sooyeon/LyCORIS/test/td_inference.txt' \
      --efficient_layer 'text,down_blocks_2,mid,up_blocks_1' \
      --trg_token td --class_token 'teddy bear'

CUDA_VISIBLE_DEVICES=4 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/perfusion_experiment/teddy_bear/te_pretrain_20_unet/last.safetensors' \
      --outdir './result/perfusion_experiment/teddy_bear/te_pretrain_20_unet/inference/epoch_16_text_proj_in_ff_net/' \
      --from_file '/data7/sooyeon/LyCORIS/test/td_inference.txt' \
      --efficient_layer 'text,proj_in,ff_net' \
      --trg_token td --class_token 'teddy bear'
#

CUDA_VISIBLE_DEVICES=1 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights "/data7/sooyeon/LyCORIS/lyco_v2/kohya_ss/result/perfusion_experiment/teddy_bear/te_pretrain_20_unet/last.safetensors" \
      --outdir './result/inference_test_teddy_bear/text_related_proj_in_ff_net' \
      --from_file '/data7/sooyeon/LyCORIS/test/td_inference.txt' \
      --efficient_layer 'text,proj_in,attn1,attn2_to_out,attn2_to_q,ff_net' \
      --trg_token td --class_token 'teddy bear'

      cp '/data7/sooyeon/LyCORIS/test/td_inference.txt' '/data7/sooyeon/LyCORIS/test/teddy_bear_inference.txt'


CUDA_VISIBLE_DEVICES=5 python gen_crossattnmap.py \
      --ckpt /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
      --network_module networks.lora \
      --network_weights './result/perfusion_experiment/teddy_bear/te_pretrain_581sen_25_unet_one_image/last.safetensors' \
      --outdir './result/inference_test_teddy_bear/te_pretrain_581sen_25_unet_one_image/overfitted_model-epoch-000064_te_proj_in_ff_net_attn2_to_v' \
      --from_file '/data7/sooyeon/LyCORIS/test/td_inference.txt' \
      --efficient_layer 'text,proj_in,ff_net,attn2_to_v' --trg_token td --class_token teddy

'