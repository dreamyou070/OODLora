#! /bin/bash

class_name="tire"
dataset_cat="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${dataset_cat}"
sub_folder="_1_res_64_up_attn2_t_2_data_11"
base_dir="../result/${dataset_cat}_experiment/${class_name}/lora_training/anormal/${sub_folder}/reconstruction"
anomaly_maps_dir="step_4_guidance_scale_8.5_start_from_origin_False_start_from_final_True_"
output_dir="metrics"

python ../lora/evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --base_dir "${base_dir}" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${anomaly_maps_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3
