import os
from PIL import Image
import numpy as np
import argparse, shutil
from random import sample


def main(args):
    base_folder = args.base_folder
    categories = os.listdir(base_folder)
    for category in categories:

        if category == args.trg_category:

            category_folder = os.path.join(base_folder, category)

            train_dir = os.path.join(category_folder, 'train')
            synthetic_dir = os.path.join(train_dir, 'bad')
            synthetic_gt_dir = os.path.join(train_dir, 'gt')
            synthetic_corrected_dir = os.path.join(train_dir, 'corrected')
            cats = os.listdir(synthetic_dir)
            for cat in cats:
                before_cat = cat
                if 'good' in cat :
                    repeat_num = 10
                else :
                    repeat_num = 1
                after_repeat_cat = f'{repeat_num}_{cat}'

                before_cat_dir = os.path.join(synthetic_dir, before_cat)
                after_repeat_cat_dir = os.path.join(synthetic_dir, after_repeat_cat)
                #print(f'{before_cat_dir} -> {after_repeat_cat_dir}')

                os.rename(before_cat_dir, after_repeat_cat_dir)

                if 'good' not in cat :

                    before_cat_dir = os.path.join(synthetic_gt_dir, before_cat)
                    after_repeat_cat_dir = os.path.join(synthetic_gt_dir, after_repeat_cat)
                    #print(f'{before_cat_dir} -> {after_repeat_cat_dir}')
                    os.rename(before_cat_dir, after_repeat_cat_dir)

                before_cat_dir = os.path.join(synthetic_corrected_dir, before_cat)
                after_repeat_cat_dir = os.path.join(synthetic_corrected_dir, after_repeat_cat)
                #print(f'{before_cat_dir} -> {after_repeat_cat_dir}')
                os.rename(before_cat_dir, after_repeat_cat_dir)

            test_dir = os.path.join(category_folder, 'test')
            synthetic_test_dir = os.path.join(test_dir, 'bad')
            synthetic_test_corrected_dir = os.path.join(test_dir, 'corrected')

            test_cats = os.listdir(synthetic_test_dir)
            for test_cat in test_cats:
                before_cat = test_cat
                after_repeat_cat = f'1_{test_cat}'

                before_cat_dir = os.path.join(synthetic_test_dir, before_cat)
                after_repeat_cat_dir = os.path.join(synthetic_test_dir, after_repeat_cat)
                #print(f'{before_cat_dir} -> {after_repeat_cat_dir}')
                os.rename(before_cat_dir, after_repeat_cat_dir)

                before_cat_dir = os.path.join(synthetic_test_corrected_dir, before_cat)
                after_repeat_cat_dir = os.path.join(synthetic_test_corrected_dir, after_repeat_cat)
                #print(f'{before_cat_dir} -> {after_repeat_cat_dir}')
                os.rename(before_cat_dir, after_repeat_cat_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
