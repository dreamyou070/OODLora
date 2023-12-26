import os
from PIL import Image
import numpy as np

base_folder = r'sample_data'
categories = os.listdir(base_folder)
for category in categories:
    if category == 'cable_gland':
        category_folder = os.path.join(base_folder, category)
        train_dir = os.path.join(category_folder, 'train')

        synthetic_dir = os.path.join(train_dir, 'synthetic_bad')
        os.makedirs(synthetic_dir, exist_ok=True)

        synthetic_gt_dir = os.path.join(train_dir, 'synthetic_gt')
        os.makedirs(synthetic_gt_dir, exist_ok=True)

        synthetic_corrected_dir = os.path.join(train_dir, 'synthetic_corrected')
        os.makedirs(synthetic_corrected_dir, exist_ok=True)

        good_folder = os.path.join(train_dir, f'bad/50_good')
        good_images = os.listdir(good_folder)
        for good_image in good_images:
            good_name, ext = os.path.splitext(good_image)
            good_image_dir = os.path.join(good_folder, good_image)
            good_image_np = np.array(Image.open(good_image_dir).resize((512,512)))

            gt_folder = os.path.join(train_dir, f'gt')
            bad_folder = os.path.join(train_dir, f'bad')
            bad_cats = os.listdir(bad_folder)
            for bad_cat in bad_cats:
                if '_' in bad_cat :
                    bad_name = bad_cat.split('_')[1]

                if 'good' not in bad_cat :

                    synthetic_cat_dir = os.path.join(synthetic_dir, bad_cat)
                    os.makedirs(synthetic_cat_dir, exist_ok=True)

                    synthetic_cat_gt_dir = os.path.join(synthetic_gt_dir, bad_cat)
                    os.makedirs(synthetic_cat_gt_dir, exist_ok=True)

                    synthetic_cat_corrected_dir = os.path.join(synthetic_corrected_dir, bad_cat)
                    os.makedirs(synthetic_cat_corrected_dir, exist_ok=True)

                    bad_cat_dir = os.path.join(bad_folder, bad_cat)
                    repeat, bad_name = bad_cat.split('_')
                    gt_cat_dir = os.path.join(gt_folder, f'{bad_name}')

                    bad_rgbs = os.listdir(bad_cat_dir)
                    bad_gts = os.listdir(gt_cat_dir)

                    for bad_rgb, bad_gt in zip(bad_rgbs, bad_gts):
                        bad_rgb_name, ext = os.path.splitext(bad_rgb)
                        bad_rgb_dir = os.path.join(bad_cat_dir, bad_rgb)
                        bad_gt_dir = os.path.join(gt_cat_dir, bad_gt)
                        bad_rgb_np = np.array(Image.open(bad_rgb_dir).resize((512,512)))
                        bad_gt_np = np.array(Image.open(bad_gt_dir).resize((512,512)).convert('RGB'))/255

                        bad_obj = bad_rgb_np * bad_gt_np
                        background = good_image_np * (1 - bad_gt_np)
                        final = (bad_obj + background).astype(np.uint8)
                        final = Image.fromarray(final)
                        final.save(os.path.join(synthetic_cat_dir, f'{bad_name}_{bad_rgb_name}_{good_name}{ext}'))

                        normal_save_dir = os.path.join(synthetic_cat_corrected_dir, f'{bad_name}_{bad_rgb_name}_{good_name}{ext}')
                        Image.open(good_image_dir).resize((512, 512)).save(normal_save_dir)

                        m = Image.open(bad_gt_dir).resize((512, 512))
                        m.save(os.path.join(synthetic_cat_gt_dir, f'{bad_name}_{bad_rgb_name}_{good_name}{ext}'))
