python image_inverting.py --device cuda:5  --process_title parksooyeon \
                          --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
                          --concept_image_folder /data7/sooyeon/MyData/perfusion_dataset/cat \
                          --prompt 'cat swimming in a pool' \
                          --negative_prompt 'low quality, worst quality, bad anatomy,bad composition, poor, low effort' \
                          --output_dir ./result/inference_result/perfusion_experiment/cat/random_init_guidance_8_ddim_50_self_mean_cat_swimming_in_a_pool --seed 42 --guidance_scale 8 \
                          --num_ddim_steps 50 --min_value 12 --max_self_input_time 20 --self_key_control

python image_inverting.py --device cuda:4  --process_title parksooyeon \
                          --pretrained_model_name_or_path /data7/sooyeon/pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
                          --concept_image_folder /data7/sooyeon/MyData/perfusion_dataset/cat \
                          --prompt 'cat wearing like a wizard' \
                          --negative_prompt 'low quality, worst quality, bad anatomy,bad composition, poor, low effort' \
                          --output_dir ./result/inference_result/perfusion_experiment/cat/random_init_guidance_8_ddim_50_self_mean_from_0_one_image --seed 42 --guidance_scale 8 \
                          --num_ddim_steps 50 --min_value 0 --max_self_input_time 10 --self_key_control


