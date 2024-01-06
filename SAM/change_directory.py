from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np
import shutil
import random
trg_ratio = 0.8
def main(args):

    print(f'step 1. prepare model')



    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        if cat == args.trg_cat :
            cat_dir = os.path.join(base_folder, f'{cat}')

            train_dir = os.path.join(cat_dir, 'train')
            test_dir = os.path.join(cat_dir, 'test')
            validation_dir = os.path.join(cat_dir, 'validation')

            train_ex_dir = os.path.join(cat_dir, 'train_ex')

            train_ex_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            train_ex_gt_dir = os.path.join(train_ex_dir, 'gt')
            os.makedirs(train_ex_rgb_dir, exist_ok=True)
            os.makedirs(train_ex_gt_dir, exist_ok=True)

            test_ex_dir = os.path.join(cat_dir, 'test_ex')

            test_ex_rgb_dir = os.path.join(test_ex_dir, 'rgb')
            test_ex_gt_dir = os.path.join(test_ex_dir, 'gt')
            os.makedirs(test_ex_rgb_dir, exist_ok=True)
            os.makedirs(test_ex_gt_dir, exist_ok=True)

            # -------------------------------------------------------------------------------------------------------
            # (1) train
            good_train_dir = os.path.join(train_dir, 'good/rgb')
            sam_train_dir = os.path.join(train_dir, f'good/gt')

            train_ex_good_rgb_dir = os.path.join(train_ex_dir, f'rgb/10_good')
            train_ex_good_gt_dir = os.path.join(train_ex_dir, f'gt/10_good')
            os.makedirs(train_ex_good_rgb_dir, exist_ok=True)
            os.makedirs(train_ex_good_gt_dir, exist_ok=True)

            images = os.listdir(good_train_dir)

            for image in images:
                org_rgb_image_path = os.path.join(good_train_dir, image)
                org_gt_image_path = os.path.join(sam_train_dir, image)
                new_rgb_image_path = os.path.join(train_ex_good_rgb_dir, f'train_{image}')
                new_gt_image_path = os.path.join(train_ex_good_gt_dir, f'train_{image}')
                Image.open(org_rgb_image_path).resize((512,512)).save(new_rgb_image_path)
                gt = Image.open(org_gt_image_path).resize((512,512)).convert('L')
                gt.save(new_gt_image_path)

            # -------------------------------------------------------------------------------------------------------
            # (2) validation
            good_validation_dir = os.path.join(validation_dir, 'good/rgb')
            sam_validation_dir = os.path.join(validation_dir, f'good/gt')
            val_images = os.listdir(good_validation_dir)
            for image in val_images:
                org_rgb_image_path = os.path.join(good_validation_dir, image)
                org_gt_image_path = os.path.join(sam_validation_dir, image)
                new_rgb_image_path = os.path.join(train_ex_good_rgb_dir, f'val_{image}')
                new_gt_image_path = os.path.join(train_ex_good_gt_dir, f'val_{image}')
                Image.open(org_rgb_image_path).resize((512,512)).save(new_rgb_image_path)
                gt = Image.open(org_gt_image_path).resize((512,512)).convert('L')
                gt.save(new_gt_image_path)

            # -------------------------------------------------------------------------------------------------------
            categories = os.listdir(test_dir)
            for category in categories:
                if 'good' not in category:
                    category_dir = os.path.join(test_dir, category)
                    train_cat_rgb_dir = os.path.join(train_ex_rgb_dir, f'50_{category}')
                    train_cat_gt_dir = os.path.join(train_ex_gt_dir, f'50_{category}')
                    os.makedirs(train_cat_rgb_dir, exist_ok=True)
                    os.makedirs(train_cat_gt_dir, exist_ok=True)
                    test_cat_rgb_dir = os.path.join(test_ex_rgb_dir, category)
                    test_cat_gt_dir = os.path.join(test_ex_gt_dir, category)
                    os.makedirs(test_cat_rgb_dir, exist_ok=True)
                    os.makedirs(test_cat_gt_dir, exist_ok=True)

                    cat_rgb_dir = os.path.join(category_dir, 'rgb')
                    cat_gt_dir = os.path.join(category_dir, 'gt')
                    cat_images = os.listdir(cat_rgb_dir)
                    num_images = len(cat_images)

                    train_index = random.sample(range(num_images), int(num_images * trg_ratio))
                    for i, img in enumerate(cat_images) :
                        org_rgb_image_path = os.path.join(cat_rgb_dir, img)
                        org_gt_image_path = os.path.join(cat_gt_dir, img)
                        org_pil_rgb_image = Image.open(org_rgb_image_path).resize((512,512))
                        org_pil_gt_image = Image.open(org_gt_image_path).convert('L').resize((512,512))

                        if i in train_index:
                            print(f'anomal to tain')
                            new_rgb_image_path = os.path.join(train_cat_rgb_dir, f'{category}_{img}')
                            new_gt_image_path = os.path.join(train_cat_gt_dir, f'{category}_{img}')

                        else :
                            new_rgb_image_path = os.path.join(test_cat_rgb_dir, f'{category}_{img}')
                            new_gt_image_path = os.path.join(test_cat_gt_dir, f'{category}_{img}')
                        org_pil_rgb_image.save(new_rgb_image_path)
                        org_pil_gt_image.save(new_gt_image_path)

                if 'good' in category :

                    category_dir = os.path.join(test_dir, category)

                    cat_rgb_dir = os.path.join(category_dir, 'rgb')
                    cat_gt_dir = os.path.join(category_dir, 'gt')
                    cat_images = os.listdir(cat_rgb_dir)

                    test_cat_rgb_dir = os.path.join(test_ex_rgb_dir, category)
                    test_cat_gt_dir = os.path.join(test_ex_gt_dir, category)
                    os.makedirs(test_cat_rgb_dir, exist_ok=True)
                    os.makedirs(test_cat_gt_dir, exist_ok=True)

                    for cat_image in cat_images :
                        org_rgb_image_path = os.path.join(cat_rgb_dir, cat_image)
                        org_gt_image_path = os.path.join(cat_gt_dir, cat_image)
                        new_rgb_image_path = os.path.join(test_cat_rgb_dir, cat_image)
                        new_gt_image_path = os.path.join(test_cat_gt_dir, cat_image)

                        Image.open(org_rgb_image_path).resize((512,512)).save(new_rgb_image_path)
                        gt = Image.open(org_gt_image_path).resize((512,512)).convert('L')
                        gt.save(new_gt_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='carrot')
    args = parser.parse_args()
    main(args)