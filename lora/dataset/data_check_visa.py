import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusers import StableDiffusionInpaintPipeline

from PIL import Image
import argparse


def main(args):

    print(f'\n step 1. dataset')
    parent, child = os.path.split(args.data_folder)
    save_folder = os.path.join(parent, f'{child}_Experiment')
    os.makedirs(save_folder, exist_ok=True)

    base_folder_dir = '/home/dreamyou070/MyData/anomaly_detection/VisA/split_csv'
    files = os.listdir(base_folder_dir)
    for file in files:
        file_dir = os.path.join(base_folder_dir, file)
        with open(file_dir, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            line_list = line.split(',')
            category = line_list[0]
            save_category_dir = os.path.join(save_folder, category)
            os.makedirs(save_category_dir, exist_ok=True)
            split = line_list[1]
            save_split_dir = os.path.join(save_category_dir, split)
            os.makedirs(save_split_dir, exist_ok=True)
            if split == 'test' :
                normal_anomal_info = line_list[2]
                final_image_dir = line_list[3].strip()
                img_name = final_image_dir.split('/')[-1]

                if normal_anomal_info == 'normal' :
                    org_img_dir = os.path.join(args.data_folder, f'{final_image_dir}')
                    normal_anomal_info_dir = os.path.join(save_split_dir, 'good')
                    os.makedirs(normal_anomal_info_dir, exist_ok=True)
                    new_img_dir = os.path.join( normal_anomal_info_dir, img_name)
                    Image.open(org_img_dir).save(new_img_dir)
                else :
                    org_img_dir = os.path.join(args.data_folder, f'{final_image_dir}')
                    normal_anomal_info_dir = os.path.join(save_split_dir, 'bad')
                    os.makedirs(normal_anomal_info_dir, exist_ok=True)
                    new_img_dir = os.path.join( normal_anomal_info_dir, img_name)
                    Image.open(org_img_dir).save(new_img_dir)

                    mask_img_dir = os.path.join(args.data_folder, f'{line_list[-1].strip()}')
                    mask_folder_dir = os.path.join(save_split_dir, 'ground_truth')
                    os.makedirs(mask_folder_dir, exist_ok=True)
                    Image.open(mask_img_dir).save(os.path.join(mask_folder_dir, img_name))

            else :
                normal_anomal_info_dir = os.path.join(save_split_dir, 'good')
                os.makedirs(normal_anomal_info_dir, exist_ok=True)
                final_image_dir = line_list[3].strip()
                img_name = final_image_dir.split('/')[-1]
                org_img_dir = os.path.join(args.data_folder, f'{final_image_dir}')
                new_img_dir = os.path.join( normal_anomal_info_dir, img_name)
                Image.open(org_img_dir).save(new_img_dir)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='../../../../MyData/anomaly_detection/VisA')
    args = parser.parse_args()
    main(args)