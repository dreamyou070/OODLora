#! /bin/bash

class_name="bagel"
dataset_cat="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${dataset_cat}"
anomaly_maps_dir="../result/${dataset_cat}/${class_name}/lora_training/anormal/2_2_res_64_up_attn2_t_2/recon_infer/2024-01-20_22-07-56"
output_dir="../result/${dataset_cat}/${class_name}/lora_training/anormal/2_2_res_64_up_attn2_t_2/recon_infer/2024-01-20_22-07-56/metrics"

python ../lora/evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${anomaly_maps_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3