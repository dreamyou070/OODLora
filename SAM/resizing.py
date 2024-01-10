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
            train_ex_gt_dir = os.path.join(train_ex_dir, 'gt')
            train_ex_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            folders = os.listdir(train_ex_rgb_dir)
            for folder in folders:
                folder_gt_dir = os.path.join(train_ex_gt_dir, folder)
                folder_rgb_dir = os.path.join(train_ex_rgb_dir, folder)
                images = os.listdir(folder_rgb_dir)
                for image in images:
                    img_gt_dir = os.path.join(folder_gt_dir, image)
                    img_rgb_dir = os.path.join(folder_rgb_dir, image)
                    Image.open(img_gt_dir).resize((512,512)).convert('L').save(img_gt_dir)
                    Image.open(img_rgb_dir).resize((512,512)).save(img_rgb_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='foam')
    args = parser.parse_args()
    main(args)