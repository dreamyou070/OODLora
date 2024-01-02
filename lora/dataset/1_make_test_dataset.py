import os
from PIL import Image
import numpy as np
import argparse, shutil
from random import sample


def main(args):

    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        if cat != 'bagel' :
            cat_folder = os.path.join(base_folder, cat)
            train_folder = os.path.join(cat_folder, 'train_ex/bad/10_good')
            corrected_folder = os.path.join(cat_folder, 'train_ex/corrected/10_good')
            images = os.listdir(train_folder)
            for image in images:
                pil = Image.fromarray(np.zeros((512,512)).astype(np.uint8))
                trg_dir = os.path.join(corrected_folder, image)
                pil.save(trg_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
