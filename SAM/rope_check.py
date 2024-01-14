from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np


def main(args):
    print(f'step 1. prepare model')

    model_type = "vit_h"
    path_to_checkpoint = r'/home/dreamyou070/pretrained_stable_diffusion/sam_vit_h_4b8939.pth'
    sam = sam_model_registry[model_type](checkpoint=path_to_checkpoint)
    predictor = SamPredictor(sam)

    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        if cat == args.trg_cat:
            cat_dir = os.path.join(base_folder, f'{cat}')
            train_ex_dir = os.path.join(cat_dir, 'train_ex')

            train_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            train_gt_dir = os.path.join(train_ex_dir, 'gt')
            train_pixel_mask_dir = os.path.join(train_ex_dir, 'mask')


            folders = os.listdir(train_pixel_mask_dir)
            for folder in folders:
                rgb_folder_dir = os.path.join(train_pixel_mask_dir, folder)
                images = os.listdir(rgb_folder_dir)
                for image in images:
                    rgb_img_dir = os.path.join(rgb_folder_dir, image)
                    np_img = np.array(Image.open(rgb_img_dir))
                    h, w, c = np_img.shape
                    print(f'h : {h} w : {w} c : {c}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='rope')
    args = parser.parse_args()
    main(args)