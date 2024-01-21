import os, json
import argparse
def main(args) :

    second_folder_name = args.second_folder_name
    condition_folder = args.condition_folder
    class_name = args.class_name

    base_save_dir = f'/home/dreamyou070/Lora/OODLora/result/MVTec3D-AD_experiment/{class_name}/lora_training/anormal/{second_folder_name}'
    total_matric_save_dir = os.path.join(base_save_dir, f'{class_name}_total_metrics.csv')

    metric_base_folder = os.path.join(base_save_dir, 'reconstruction')
    lora_folders = os.listdir(metric_base_folder)

    total_metrics = []
    title = ['lora_folder', 'au_pro', 'au_roc', 'roc_curve_fpr', '','','roc_curve_tpr','','']
    total_metrics.append(title)
    for lora_folder in lora_folders:
        lora_dir = os.path.join(metric_base_folder, lora_folder)
        metric_dir = os.path.join(lora_dir, f'{condition_folder}/metrics/metrics.json')
        with open(metric_dir, 'r') as f:
            content = json.load(f)
        metric = content[args.class_name]
        pro = metric['au_pro']
        roc = metric['au_roc']
        elem = [lora_folder, pro, roc]
        roc_curve_fpr = metric['roc_curve_fpr']
        for i in range(len(roc_curve_fpr)):
            elem.append(roc_curve_fpr[i])
        roc_curve_tpr = metric['roc_curve_tpr']
        for i in range(len(roc_curve_tpr)):
            elem.append(roc_curve_tpr[i])
        total_metrics.append(elem)

    with open(total_matric_save_dir, 'w') as f:
        for elem in total_metrics:
            for i in range(len(elem)):
                if i == len(elem) - 1:
                    f.write(str(elem[i]) + '\n')
                else:
                    f.write(str(elem[i]) + ',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default='carrot')
    parser.add_argument('--condition_folder', type=str, default='step_4_guidance_scale_8.5_start_from_origin_False_start_from_final_True_')
    parser.add_argument('--second_folder_name', type=str, default='2_2_res_64_up_attn2_t_2_20240121')

    args = parser.parse_args()
    main(args)