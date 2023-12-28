import os
from PIL import Image
import numpy as np
import argparse, shutil
from random import sample


def main(args):

    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        if cat == 'cable_gland' :
            cat_folder = os.path.join(base_folder, cat)

            test_ex_folder = os.path.join(cat_folder, 'test_ex')
            test_ex_bad_folder = os.path.join(test_ex_folder, 'bad')
            test_ex_gt_folder = os.path.join(test_ex_folder, 'corrected')

            train_ex_folder = os.path.join(cat_folder, 'train_ex')
            train_ex_bad_folder = os.path.join(train_ex_folder, 'bad')
            train_ex_gt_folder = os.path.join(train_ex_folder, 'corrected')

            cats = os.listdir(test_ex_bad_folder)
            for cat in cats:
                org_cat_folder = os.path.join(test_ex_bad_folder, cat)
                new_cat_folder = os.path.join(test_ex_bad_folder, f'1_{cat}')
                os.rename(org_cat_folder, new_cat_folder)

                org_gt_folder = os.path.join(test_ex_gt_folder, cat)
                new_gt_folder = os.path.join(test_ex_gt_folder, f'1_{cat}')
                os.rename(org_gt_folder, new_gt_folder)

            train_cats = os.listdir(train_ex_bad_folder)
            for cat in train_cats:
                org_cat_folder = os.path.join(train_ex_bad_folder, cat)
                new_cat_folder = os.path.join(train_ex_bad_folder, f'10_{cat}')
                os.rename(org_cat_folder, new_cat_folder)

                org_gt_folder = os.path.join(train_ex_gt_folder, cat)
                new_gt_folder = os.path.join(train_ex_gt_folder, f'10_{cat}')
                os.rename(org_gt_folder, new_gt_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
