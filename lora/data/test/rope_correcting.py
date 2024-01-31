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
            train_ex_dir = os.path.join(cat_dir, 'train_normal')

            train_ex_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            train_ex_gt_dir = os.path.join(train_ex_dir, 'gt')
            test_ex_rgb_dir = os.path.join(train_ex_dir, '')
            os.makedirs(test_ex_rgb_dir, exist_ok=True)

            folders = os.listdir(train_ex_gt_dir)

            for folder in folders:
                folder_dir = os.path.join(train_ex_gt_dir, folder)
                test_folder_dir = os.path.join(test_ex_rgb_dir, folder)
                os.makedirs(test_folder_dir, exist_ok=True)

                images = os.listdir(folder_dir)
                for image in images:
                    image_dir = os.path.join(folder_dir, image)
                    pil = Image.open(image_dir)
                    #h, w = pil.size
                    img_np = np.array(pil)
                    h, w = img_np.shape
                    for i in range(h):
                        if i < h* (14/40) or i > h * (27/40):
                            img_np[i,:] = 0
                    Image.fromarray(img_np.astype(np.uint8)).save(os.path.join(test_folder_dir, image))


# 20 + 120 = 140
# 40 + 160 = 200
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='rope')
    parser.add_argument('--new_ok_repeat', type=int, default=20)
    parser.add_argument('--new_nok_repeat', type=int, default=20)
    args = parser.parse_args()
    main(args)