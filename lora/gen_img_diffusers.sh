cuda_visible_devices=6 python generate_crossattn_map.py \
                    --network_module networks.lora \
                    --ckpt /data7/sooyeon/LyCORIS/lyco_v2/kohya_ss/result/jungwoo_textual_inversion_experience/jungwoo_textual_inversion_1/jungwoo.safetensors \
                    --network_weights "./result/jungwoo_experience/jungwoo_3_blur_mask_only_down_blocks_0_attentions_1_countinue/jungwoo-000024.safetensors" \
                    --outdir ./test/jungwoo_3_blur_mask_only_down_blocks_0_attentions_1_countinue/jw\
                    --seed 42 \
                    --prompt "jw, white_background, standing, smile, male_focused" --trg_token jw \
                    --H 512 --W 512 --clip_skip 2
                     \
                    # --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo.txt

      --prompt "jw, white_background" --outdir 20231008_result/jungwoo_4_crossattn_calculate_change_only_up --seed 42 --trg_token 'jw'

cuda_visible_devices=6 python gen_img_diffusers.py \
                    --ckpt /data7/sooyeon/LyCORIS/lyco_v2/kohya_ss/result/jungwoo_textual_inversion_experience/jungwoo_textual_inversion_1/jungwoo.safetensors \
                    --outdir ./test/jungwoo_textual_inversion_experience/jungwoo\
                    --seed 42 --H 512 --W 512 --clip_skip 2 \
                    --prompt "jungwoo, white_background, standing, smile, male_focused" --trg_token jungwoo

                     \
                    # --from_file /data7/sooyeon/LyCORIS/LyCORIS/test/test_jungwoo.txt

      --prompt "jw, white_background" --outdir 20231008_result/jungwoo_4_crossattn_calculate_change_only_up --seed 42 --trg_token 'jw'