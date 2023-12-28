import os
from PIL import Image
import numpy as np
import argparse, shutil
from random import sample


def main(args):

    base_folder = args.base_folder
    categories = os.listdir(base_folder)
    for category in categories:
        category_folder = os.path.join(base_folder, category)

        train_folder = os.path.join(category_folder, 'train')
        test_ex_folder = os.path.join(category_folder, 'test_ex')
        os.makedirs(test_ex_folder, exist_ok=True)
        test_bad_folder = os.path.join(test_ex_folder, 'bad')
        os.makedirs(test_bad_folder, exist_ok=True)
        test_gt_folder = os.path.join(test_ex_folder, 'corrected')
        os.makedirs(test_gt_folder, exist_ok=True)

        test_folder = os.path.join(category_folder, 'test')
        train_ex_folder = os.path.join(category_folder, 'train_ex')
        os.makedirs(train_ex_folder, exist_ok=True)
        train_bad_folder = os.path.join(train_ex_folder, 'bad')
        os.makedirs(train_bad_folder, exist_ok=True)
        train_gt_folder = os.path.join(train_ex_folder, 'corrected')
        os.makedirs(train_gt_folder, exist_ok=True)

        categories = os.listdir(test_folder)

        for category in categories:

            if 'good' not in category:

                category_folder = os.path.join(test_folder, category)

                train_cat_bad_folder = os.path.join(train_bad_folder, category)
                os.makedirs(train_cat_bad_folder, exist_ok=True)
                train_cat_gt_folder = os.path.join(train_gt_folder, category)
                os.makedirs(train_cat_gt_folder, exist_ok=True)

                test_cat_bad_folder = os.path.join(test_bad_folder, category)
                os.makedirs(test_cat_bad_folder, exist_ok=True)
                test_cat_gt_folder = os.path.join(test_gt_folder, category)
                os.makedirs(test_cat_gt_folder, exist_ok=True)

                rgb_folder = os.path.join(category_folder, 'rgb')
                gt_folder = os.path.join(category_folder, 'gt')

                images = os.listdir(rgb_folder)
                num_images = len(images)
                train_num = int(num_images * 0.8)
                for i in range(num_images):
                    image = images[i]
                    rgb_path = os.path.join(rgb_folder, image)
                    gt_path = os.path.join(gt_folder, image)

                    if i < train_num :

                        new_rgb_path = os.path.join(train_cat_bad_folder, image)
                        new_gt_path = os.path.join(train_cat_gt_folder, image)
                        shutil.copy(rgb_path, new_rgb_path)
                        shutil.copy(gt_path, new_gt_path)

                    else :
                        new_rgb_path = os.path.join(test_cat_bad_folder, image)
                        new_gt_path = os.path.join(test_cat_gt_folder, image)
                        shutil.copy(rgb_path, new_rgb_path)
                        shutil.copy(gt_path, new_gt_path)

            if 'good' in category:

                category_folder = os.path.join(test_folder, category)

                train_cat_bad_folder = os.path.join(train_bad_folder, category)
                os.makedirs(train_cat_bad_folder, exist_ok=True)
                train_cat_gt_folder = os.path.join(train_gt_folder, category)
                os.makedirs(train_cat_gt_folder, exist_ok=True)

                test_cat_bad_folder = os.path.join(test_bad_folder, category)
                os.makedirs(test_cat_bad_folder, exist_ok=True)
                test_cat_gt_folder = os.path.join(test_gt_folder, category)
                os.makedirs(test_cat_gt_folder, exist_ok=True)

                rgb_folder = os.path.join(category_folder, 'rgb')
                os.makedirs(rgb_folder, exist_ok=True)
                gt_folder = os.path.join(category_folder, 'gt')
                os.makedirs(gt_folder, exist_ok=True)

                shutil.copytree(rgb_folder, test_cat_bad_folder)
                shutil.copytree(gt_folder, test_cat_gt_folder)

        train_good_rgb_folder = os.path.join(train_folder,f'good/rgb' )
        new_train_good_folder = os.path.join(train_bad_folder, 'good')
        shutil.copytree(train_good_rgb_folder, new_train_good_folder)

        new_train_gt_folder = os.path.join(train_gt_folder, 'good')
        os.makedirs(new_train_gt_folder, exist_ok=True)
        train_images = os.listdir(train_good_rgb_folder)
        for image in train_images:
            save_dir = os.path.join(new_train_gt_folder, image)
            pil = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
            pil.save(save_dir)

        validation_folder = os.path.join(category_folder, 'validation/good/rgb')
        val_images = os.listdir(validation_folder)
        for image in val_images:
            val_pil = Image.open(os.path.join(validation_folder, image))
            val_pil.save(os.path.join(new_train_good_folder, f'val_{image}'))
            msk_pil = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
            msk_pil.save(os.path.join(new_train_gt_folder, f'val_{image}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
