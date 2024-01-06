import os
from PIL import Image
import numpy as np
import argparse, shutil
from random import sample


def main(args):

    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        if cat != 'bagel' :
            cat_folder = os.path.join(base_folder, cat)
            train_folder = os.path.join(cat_folder, 'train')
            test_folder = os.path.join(cat_folder, 'test')
            validation_folder = os.path.join(cat_folder, 'validation')

            train_ex_folder = os.path.join(cat_folder, 'train_ex')
            test_ex_folder = os.path.join(cat_folder, 'test_ex')
            os.makedirs(train_ex_folder, exist_ok=True)
            os.makedirs(test_ex_folder, exist_ok=True)

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
