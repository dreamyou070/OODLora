import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import argparse
import shutil

def main(args):

    print(f'\n step 1. make model')
    #device = args.device


    print(f'\n step 2. dataset')
    base_folder = '../../../../MyData/anomaly_detection/MVTec3D-AD'

    data_folder = args.data_folder
    classes = os.listdir(data_folder)

    for cls in classes:

        if cls == 'cable_gland' :

            cls_folder = os.path.join(data_folder, cls)
            train_folder = os.path.join(cls_folder, 'train')
            bad_folder = os.path.join(train_folder, 'bad')
            corrected_folder = os.path.join(train_folder, 'corrected')
            masked_folder = os.path.join(train_folder, 'gt')
            os.makedirs(corrected_folder, exist_ok=True)
            cats = os.listdir(corrected_folder)
            for cat in cats:
                if 'good' not in cat :
                    cat_folder = os.path.join(bad_folder, cat)
                    inpainted_folder = os.path.join(corrected_folder, cat)
                    os.makedirs(inpainted_folder, exist_ok=True)
                    images = os.listdir(cat_folder)
                    for image in images:
                        name, ext = image.split('.')
                        if 'inpainted' in name:
                            img_dir = os.path.join(cat_folder, image)
                            number = name.split('_')[0]
                            new_img_dir = os.path.join(inpainted_folder,f'{number}.{ext}')
                            os.rename(img_dir, new_img_dir)
                        if 'mask' in name:
                            img_dir = os.path.join(cat_folder, image)
                            number = name.split('_')[0]
                            new_img_dir = os.path.join(masked_folder,f'{number}.{ext}')
                            os.rename(img_dir, new_img_dir)
                else :
                    cat_folder = os.path.join(bad_folder, cat)
                    new_cat_folder = os.path.join(corrected_folder, cat)
                    shutil.copytree(cat_folder, new_cat_folder)

            test_folder = os.path.join(cls_folder, 'test')
            rgb_folder = os.path.join(test_folder, 'rgb')
            gt_dir = os.path.join(test_folder, 'gt')
            os.makedirs(gt_dir, exist_ok=True)
            cats = os.listdir(rgb_folder)
            for cat in cats:
                cat_dir = os.path.join(test_folder, cat)
                images = os.listdir(cat_dir)
                for image in images:
                    img_dir = os.path.join(cat_dir, image)
                    if 'inpainted' in image:
                        os.remove(img_dir)
                    elif 'mask' in image:
                        gt_cat_dir = os.path.join(gt_dir, cat)
                        os.makedirs(gt_cat_dir, exist_ok=True)
                        number, ext = image.split('.')
                        number = number.split('_')[0]
                        new_img_dir = os.path.join(gt_cat_dir, f'{number}.{ext}')
                        os.rename(img_dir, new_img_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='../../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL')
    args = parser.parse_args()
    main(args)
