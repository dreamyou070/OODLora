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
            train_ex_dir = os.path.join(cat_dir, 'test_ex')

            train_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            train_gt_dir = os.path.join(train_ex_dir, 'gt')
            train_pixel_mask_dir = os.path.join(train_ex_dir, 'mask')
            re_train_pixel_mask_dir = os.path.join(train_ex_dir, 'mask_re')
            os.makedirs(re_train_pixel_mask_dir, exist_ok=True)

            folders = os.listdir(train_pixel_mask_dir)
            for folder in folders:
                if 'good' in folder:
                    rgb_folder_dir = os.path.join(train_pixel_mask_dir, folder)
                    re_rgb_folder_dir = os.path.join(re_train_pixel_mask_dir, folder)
                    os.makedirs(re_rgb_folder_dir, exist_ok=True)
                    images = os.listdir(rgb_folder_dir)
                    for image in images:
                        rgb_img_dir = os.path.join(rgb_folder_dir, image)
                        np_img = np.array(Image.open(rgb_img_dir))
                        h, w = np_img.shape
                        min_h , max_h = int(h/3), int(h*2/3)
                        for h_index in range(h) :
                            if h_index < min_h or h_index > max_h :
                                np_img[h_index, :] = 0
                        new_mask = Image.fromarray(np_img)
                        new_mask.save(os.path.join(re_rgb_folder_dir, image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='rope')
    args = parser.parse_args()
    main(args)