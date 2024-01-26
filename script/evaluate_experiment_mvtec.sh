#! /bin/bash

class_name="cable_gland"
dataset_cat="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${dataset_cat}"
sub_folder="res_64_up_attn2_t_2_attn2_data_12"
base_dir="../result/${dataset_cat}_experiment/${class_name}/lora_training/anormal/${sub_folder}/reconstruction"
anomaly_maps_dir="detec_epoch_4_anomal_thredhold_0.5_step_4_guidance_8.5"

output_dir="metrics"

python ../lora/evaluation/evaluation_code_MVTec/evaluate_experiment_2.py \
     --base_dir "${base_dir}" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${anomaly_maps_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3