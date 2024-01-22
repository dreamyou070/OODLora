from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np
import shutil

def main(args):

    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        cat_dir = os.path.join(base_folder, f'{cat}')
        train_normal_dir = os.path.join(cat_dir, 'train_normal')
        folders = os.listdir(train_normal_dir)
        for folder in folders:
            folder_dir = os.path.join(train_normal_dir, f'{folder}')
            sub_folders = os.listdir(folder_dir)
            for sub_folder in sub_folders:
                sub_folder_dir = os.path.join(folder_dir, f'{sub_folder}')
                if 'good' not in sub_folder_dir:
                    shutil.rmtree(sub_folder_dir)
                else :
                    r_num, name = sub_folder.split('_')
                    new_name = f'40_{name}'
                    new_sub_folder_dir = os.path.join(folder_dir, f'{new_name}')
                    os.rename(sub_folder_dir, new_sub_folder_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='foam')
    args = parser.parse_args()
    main(args)