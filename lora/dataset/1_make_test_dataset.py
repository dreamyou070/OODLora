import os
from PIL import Image
import numpy as np
import argparse, shutil
from random import sample


def main(args):

    base_folder = args.base_folder
    categories = os.listdir(base_folder)
    for category in categories:
        category_folder = os.path.join(base_folder, category)
        test_folder = os.path.join(category_folder, 'test')
        folders = os.listdir(test_folder)
        for folder in folders:
            if '_' not in folder and folder != 'bad' :
                folder_dir = os.path.join(test_folder, folder)
                rgb_dir = os.path.join(folder_dir, 'rgb')
                gt_dir = os.path.join(folder_dir, 'gt')
                new_rgb_dir = os.path.join(test_folder, f'bad/1_{folder}')
                new_gt_dir = os.path.join(test_folder, f'corrected/1_{folder}')
                shutil.copytree(rgb_dir, new_rgb_dir)
                shutil.copytree(gt_dir, new_gt_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
