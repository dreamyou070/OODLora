#! /bin/bash

class_name="tire"
condition_folder='step_4_guidance_scale_8.5_start_from_origin_False_start_from_final_True_'
second_folder_name=''

python ../lora/evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --condition_folder ${condition_folder} \
  --second_folder_name ${second_folder_name}
