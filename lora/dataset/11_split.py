import os
from PIL import Image
import numpy as np
import argparse
from random import sample
def main(args) :

    base_folder = args.base_folder
    categories = os.listdir(base_folder)
    for category in categories:
        if category == args.trg_category:

            category_folder = os.path.join(base_folder, category)
            train_dir = os.path.join(category_folder, 'train')
            good_img_dir = os.path.join(train_dir, 'good/rgb')
            good_imges = os.listdir(good_img_dir)
            test_dir = os.path.join(category_folder, 'test')

            synthetic_dir = os.path.join(train_dir, 'bad')
            synthetic_gt_dir = os.path.join(train_dir, 'gt')
            synthetic_corrected_dir = os.path.join(train_dir, 'corrected')

            synthetic_test_dir = os.path.join(test_dir, 'bad')
            synthetic_test_gt_dir = os.path.join(test_dir, 'gt')
            synthetic_test_corrected_dir = os.path.join(test_dir, 'corrected')
            os.makedirs(synthetic_dir, exist_ok=True)
            os.makedirs(synthetic_gt_dir, exist_ok=True)
            os.makedirs(synthetic_corrected_dir, exist_ok=True)

            bad_cats = os.listdir(test_dir)
            for bad_cat in bad_cats :
                bad_cat_dir = os.path.join(test_dir, bad_cat)
                bad_cat_rgb_folder = os.path.join(bad_cat_dir, 'rgb')
                images = os.listdir(bad_cat_rgb_folder)

                new_long_name_rgb = os.path.join(synthetic_test_dir, f'{bad_cat}')
                new_long_name_corrected = os.path.join(synthetic_test_corrected_dir, f'{bad_cat}')
                new_long_name_gt = os.path.join(synthetic_test_gt_dir, f'{bad_cat}')
                os.makedirs(new_long_name_rgb, exist_ok=True)
                os.makedirs(new_long_name_corrected, exist_ok=True)
                os.makedirs(new_long_name_gt, exist_ok=True)

                new_names = []
                for image in images:
                    name, ext = os.path.splitext(image)
                    new_name = f'{bad_cat}_{name}'
                    new_names.append(new_name)

                num_imgs = len(new_names)
                random_idx = sample(range(0, num_imgs), int(num_imgs * 0.2))
                for idx in random_idx:
                    image = images[idx]
                    for good_img in good_imges:
                        good_img_pure_name, ext = os.path.splitext(good_img)
                        long_name = f'{image}_{good_img_pure_name}'

                        long_name_rgb_dir = os.path.join(synthetic_dir, f'{bad_cat}/{long_name}')
                        long_name_corrected_dir = os.path.join(synthetic_corrected_dir, f'{bad_cat}/{long_name}')
                        long_name_gt_dir  = os.path.join(synthetic_gt_dir, f'{bad_cat}/{long_name}')
                        new_long_name_rgb_dir = os.path.join(synthetic_test_dir, f'{bad_cat}/{long_name}')
                        new_long_name_corrected_dir = os.path.join(synthetic_test_corrected_dir, f'{bad_cat}/{long_name}')
                        new_long_name_gt_dir  = os.path.join(synthetic_test_gt_dir, f'{bad_cat}/{long_name}')
                        os.rename(long_name_rgb_dir, new_long_name_rgb_dir)
                        os.rename(long_name_corrected_dir, new_long_name_corrected_dir)
                        os.rename(long_name_gt_dir, new_long_name_gt_dir)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
