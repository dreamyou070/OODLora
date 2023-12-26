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
            test_dir = os.path.join(category_folder, 'test')

            synthetic_dir = os.path.join(train_dir, 'bad')
            synthetic_gt_dir = os.path.join(train_dir, 'gt')
            synthetic_corrected_dir = os.path.join(train_dir, 'corrected')

            syn_test_dir = os.path.join(test_dir, 'synthetic_bad')
            os.makedirs(syn_test_dir, exist_ok=True)
            syn_test_gt_dir = os.path.join(test_dir, 'synthetic_gt')
            os.makedirs(syn_test_gt_dir, exist_ok=True)
            syn_test_corrected_dir = os.path.join(test_dir, 'synthetic_corrected')
            os.makedirs(syn_test_corrected_dir, exist_ok=True)

            test_folder = os.path.join(category_folder, f'test')

            cats = os.listdir(synthetic_dir)
            for cat in cats:

                cat_synthetic_dir = os.path.join(synthetic_dir, cat)
                cat_synthetic_gt_dir = os.path.join(synthetic_gt_dir, cat)
                cat_synthetic_corrected_dir = os.path.join(synthetic_corrected_dir, cat)

                cat_syn_test_dir = os.path.join(syn_test_dir, cat)
                os.makedirs(cat_syn_test_dir, exist_ok=True)
                cat_syn_test_gt_dir = os.path.join(syn_test_gt_dir, cat)
                os.makedirs(cat_syn_test_gt_dir, exist_ok=True)
                cat_syn_test_corrected_dir = os.path.join(syn_test_corrected_dir, cat)
                os.makedirs(cat_syn_test_corrected_dir, exist_ok=True)


                images = os.listdir(cat_synthetic_dir)
                num_imgs = len(images)
                random_idx = sample(range(0, num_imgs), int(num_imgs * 0.2))
                for idx in random_idx:
                    image = images[idx]
                    bad_image_dir = os.path.join(cat_synthetic_dir, image)
                    good_image_dir = os.path.join(cat_synthetic_gt_dir, image)
                    corrected_image_dir = os.path.join(cat_synthetic_corrected_dir, image)

                    re_bad_image_dir = os.path.join(cat_syn_test_dir, image)
                    re_good_image_dir = os.path.join(cat_syn_test_gt_dir, image)
                    re_corrected_image_dir = os.path.join(cat_syn_test_corrected_dir, image)
                    os.rename(bad_image_dir, re_bad_image_dir)
                    os.rename(corrected_image_dir, re_corrected_image_dir)
                    if 'good' in cat :
                        Image.fromarray(np.zeros((512, 512)).astype(np.uint8)).save(re_corrected_image_dir)
                    else :
                        os.rename(good_image_dir, re_good_image_dir)





if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
