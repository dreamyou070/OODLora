import os
from PIL import Image
import numpy as np
import argparse
import shutil
def main(args) :

    base_folder = args.base_folder
    categories = os.listdir(base_folder)
    for category in categories:
        if category != 'erase' and category != 'carpet':

            category_folder = os.path.join(base_folder, category)

            train_dir = os.path.join(category_folder, 'train')
            test_folder = os.path.join(category_folder, f'test')

            good_folder = os.path.join(train_dir, 'good/rgb')

            new_good_folder = os.path.join(train_dir, 'bad/good')
            shutil.copytree(good_folder, new_good_folder)
            new_good_corrected_folder = os.path.join(train_dir, 'corrected/good')
            shutil.copytree(good_folder, new_good_corrected_folder)





if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
