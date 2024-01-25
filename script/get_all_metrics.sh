#! /bin/bash

class_name="carrot"
condition_folder='detec_epoch_3_anomal_thredhold_0.5_step_4_guidance_8.5'
second_folder_name="res_64_up_attn2_t_2_attn2"

python ../lora/evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --condition_folder ${condition_folder} \
  --second_folder_name ${second_folder_name}
