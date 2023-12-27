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

            bad_folder = os.path.join(base_folder, f'bagel/train/bad')
            corrected_folder = os.path.join(base_folder, f'bagel/train/corrected')
            cats = os.listdir(bad_folder)
            for cat in cats:

                bad_cat_folder = os.path.join(bad_folder, cat)
                corrected_cat_folder = os.path.join(corrected_folder, cat)

                images = os.listdir(corrected_cat_folder)

                for image in images:

                    bad_dir = os.path.join(bad_cat_folder, image)

                    if not os.path.exists(bad_dir):

                        name, ext = os.path.splitext(image)
                        name_list = name.split('_')
                        bad_pure_name = f'{name_list[1]}{ext}'
                        test_dir = os.path.join(base_folder, f'{category}/test/{cat}/rgb')
                        bad_source_dir = os.path.join(test_dir, bad_pure_name)
                        os.copy(bad_source_dir, bad_dir)



if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
