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

            train_ex_dir = os.path.join(cat_dir, 'train_ex')
            train_ex_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            train_ex_gt_dir = os.path.join(train_ex_dir, 'gt')

            folders = os.listdir(train_ex_rgb_dir)
            for folder in folders:
                repeat, name = folder.split('_')
                if 'good' not in name:
                    new_repeat = args.new_nok_repeat
                if 'good' in name:
                    new_repeat = args.new_ok_repeat
                new_folder = f'{new_repeat}_{name}'
                org_folder = os.path.join(train_ex_rgb_dir, folder)
                new_folder = os.path.join(train_ex_rgb_dir, new_folder)
                org_gt_folder = os.path.join(train_ex_gt_dir, folder)
                new_gt_folder = os.path.join(train_ex_gt_dir, new_folder)
                os.rename(org_folder, new_folder)
                os.rename(org_gt_folder, new_gt_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD-org')
    parser.add_argument('--trg_cat', type=str, default='carrot')
    parser.add_argument('--new_ok_repeat', type=int, default=60)
    parser.add_argument('--new_nok_repeat', type=int, default=60)
    args = parser.parse_args()
    main(args)