#! /bin/bash

class_name="bagel"
dataset_cat="MVTec3D-AD"
dataset_dir="/home/dreamyou070/MyData/anomaly_detection/${dataset_cat}/${class_name}"
anomaly_maps_dir="/home/dreamyou070/Lora/OODLora/result/${dataset_cat}/${class_name}/lora_training/anormal/2_2_res_64_up_attn2_t_2/recon_infer/2024-01-20_21-09-26"
output_dir="metrics"

python evaluate_experiment.py --dataset_base_dir "${dataset_dir}" \
                              --anomaly_maps_dir "${anomaly_maps_dir}" \
                              --output_dir "${output_dir}" \
                              --pro_integration_limit 0.3