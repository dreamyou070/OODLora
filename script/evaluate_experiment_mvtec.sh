#! /bin/bash

class_name="bagle"
dataset_cat="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${dataset_cat}"
sub_folder="res_64_up_attn12_from_normal_self_cross_attn"
base_dir="../result/${dataset_cat}_experiment/${class_name}/lora_training/anormal/${sub_folder}/reconstruction"
anomaly_maps_dir="step_4_guidance_scale_8.5_start_from_origin_False_start_from_final_True_"
output_dir="metrics"

python ../lora/evaluation/evaluation_code_MVTec/evaluate_experiment_2.py \
     --base_dir "${base_dir}" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${anomaly_maps_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3