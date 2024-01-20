#! /bin/bash

dataset_dir='/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/bagel'
anomaly_maps_dir='/home/dreamyou070/Lora/OODLora/evaluate/MVTec3D-AD'
output_dir='metrics'

python evaluate_experiment.py --dataset_base_dir '${dataset_dir}' \
                              --anomaly_maps_dir 'path/to/anomaly_maps/' \
                              --output_dir '${output_dir}' \'
                              --pro_integration_limit 0.3