import os
from PIL import Image
import numpy as np
import argparse
def main(args) :

    base_folder = args.base_folder
    categories = os.listdir(base_folder)
    for category in categories:
        if category == args.trg_category:
            category_folder = os.path.join(base_folder, category)

            train_dir = os.path.join(category_folder, 'train')

            synthetic_dir = os.path.join(train_dir, 'bad')
            os.makedirs(synthetic_dir, exist_ok=True)
            synthetic_gt_dir = os.path.join(train_dir, 'gt')
            os.makedirs(synthetic_gt_dir, exist_ok=True)
            synthetic_corrected_dir = os.path.join(train_dir, 'corrected')
            os.makedirs(synthetic_corrected_dir, exist_ok=True)

            test_folder = os.path.join(category_folder, f'test')
            bad_cats = os.listdir(test_folder)

            good_folder = os.path.join(train_dir, f'good/rgb')
            good_images = os.listdir(good_folder)
            for good_image in good_images:
                good_name, ext = os.path.splitext(good_image)
                good_image_dir = os.path.join(good_folder, good_image)
                good_image_np = np.array(Image.open(good_image_dir).resize((512,512)))

                for bad_cat in bad_cats :
                    bad_cat_dir = os.path.join(test_folder, bad_cat)
                    bad_cat_rgb_folder = os.path.join(bad_cat_dir, 'rgb')
                    bad_cat_gt_folder = os.path.join(bad_cat_dir, 'gt')

                    synthetic_cat_dir = os.path.join(synthetic_dir, bad_cat)
                    os.makedirs(synthetic_cat_dir, exist_ok=True)
                    synthetic_cat_gt_dir = os.path.join(synthetic_gt_dir, bad_cat)
                    os.makedirs(synthetic_cat_gt_dir, exist_ok=True)
                    synthetic_cat_corrected_dir = os.path.join(synthetic_corrected_dir, bad_cat)
                    os.makedirs(synthetic_cat_corrected_dir, exist_ok=True)

                    bad_rgbs = os.listdir(bad_cat_rgb_folder)
                    bad_gts = os.listdir(bad_cat_gt_folder)
                    for bad_rgb, bad_gt in zip(bad_rgbs, bad_gts):
                        bad_rgb_name, ext = os.path.splitext(bad_rgb)

                        bad_rgb_dir = os.path.join(bad_cat_rgb_folder, bad_rgb)
                        bad_gt_dir = os.path.join(bad_cat_gt_folder, bad_gt)
                        bad_rgb_np = np.array(Image.open(bad_rgb_dir).resize((512,512)))
                        bad_gt_np = np.array(Image.open(bad_gt_dir).resize((512,512)).convert('RGB'))/255

                        bad_obj = bad_rgb_np * bad_gt_np
                        background = good_image_np * (1 - bad_gt_np)
                        final = (bad_obj + background).astype(np.uint8)
                        final = Image.fromarray(final)

                        final.save(os.path.join(synthetic_cat_dir, f'{bad_cat}_{bad_rgb_name}_{good_name}{ext}'))

                        normal_save_dir = os.path.join(synthetic_cat_corrected_dir, f'{bad_cat}_{bad_rgb_name}_{good_name}{ext}')
                        Image.open(good_image_dir).resize((512, 512)).save(normal_save_dir)

                        if 'good' in bad_cat :
                            m = Image.open(bad_gt_dir).resize((512, 512))
                            m.save(os.path.join(synthetic_cat_gt_dir, f'{bad_cat}_{bad_rgb_name}_{good_name}{ext}'))
                        else :
                            Image.open(bad_gt_dir).resize((512, 512)).save(os.path.join(synthetic_cat_gt_dir, f'{bad_cat}_{bad_rgb_name}_{good_name}{ext}'))
                        break
                    break
                break


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_category', type=str, default='bagel')
    args = parser.parse_args()
    main(args)
