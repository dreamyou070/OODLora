from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np


def main(args):

    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        if cat == args.trg_cat:

            cat_dir = os.path.join(base_folder, f'{cat}')
            print(f'cat_dir: {cat_dir}')

            train_ex_dir = os.path.join(cat_dir, 'train_ex')
            print(f'train_ex_dir: {train_ex_dir}')
            train_ex_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            train_ex_gt_dir = os.path.join(train_ex_dir, 'gt')
            train_ex_mask_dir = os.path.join(train_ex_dir, 'mask')
            folders = os.listdir(train_ex_rgb_dir)

            for folder in folders:
                repeat, name = folder.split('_')
                org_rgb_folder = os.path.join(train_ex_rgb_dir, folder)
                org_gt_folder  = os.path.join(train_ex_gt_dir, folder)
                org_mask_folder = os.path.join(train_ex_mask_dir, folder)
                if 'good' in name:
                    new_repeat = args.new_ok_repeat
                else:
                    new_repeat = args.new_nok_repeat
                new_folder = f'{new_repeat}_{name}'
                print(new_folder)

                new_rgb_folder = os.path.join(train_ex_rgb_dir, new_folder)
                new_gt_folder = os.path.join(train_ex_gt_dir, new_folder)
                new_mask_folder = os.path.join(train_ex_mask_dir, new_folder)

                os.rename(org_rgb_folder, new_rgb_folder)
                os.rename(org_gt_folder, new_gt_folder)
                os.rename(org_mask_folder, new_mask_folder)

# 20 + 120 = 140
# 40 + 160 = 200
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='carrot')
    parser.add_argument('--new_ok_repeat', type=int, default=20)
    parser.add_argument('--new_nok_repeat', type=int, default=20)
    args = parser.parse_args()
    main(args)