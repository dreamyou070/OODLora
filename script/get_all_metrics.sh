#! /bin/bash

class_name="rope"
condition_folder='step_4_guidance_scale_8.5_start_from_origin_False_start_from_final_True_'
second_folder_name='2_2_res_64_up_attn2_t_2_2024012'

NCCL_P2P_DISABLE=1 accelerate launch --config_file ../../../gpu_config/gpu_0_config \
  --main_process_port ${port_number} ../lora/evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --condition_folder ${condition_folder} \
  --second_folder_name ${second_folder_name}