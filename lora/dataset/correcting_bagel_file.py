import os
from PIL import Image
import numpy as np
import argparse
from random import sample
def main(args) :

    base_folder = args.base_folder
    categories = os.listdir(base_folder)
    for category in categories:
        if category == 'bagel' :

            train_gt_dir = os.path.join(base_folder, f'bagel/train/gt')
            folders = os.listdir(train_gt_dir)
            for folder in folders :
                if '_' not in folder :
                    train_folder = os.path.join(train_gt_dir, folder)
                    test_folder = os.path.join(train_gt_dir, f'test_{folder}')
                    test_imgs = os.listdir(test_folder)
                    for test_img in test_imgs :
                        org_dir = os.path.join(test_folder, test_img)
                        new_dir = os.path.join(train_folder, test_img)
                        os.rename(org_dir, new_dir)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
