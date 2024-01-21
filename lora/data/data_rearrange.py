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

            train_dir = os.path.join(cat_dir, 'train')

            print(f'make train ex dir')

            train_ex_dir = os.path.join(train_dir, 'train_ex')
            os.makedirs(train_ex_dir, exist_ok=True)
            train_good_dir = os.path.join(train_dir, 'good')

            train_ex_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            os.makedirs(train_ex_rgb_dir, exist_ok=True)
            train_ex_gt_dir = os.path.join(train_ex_dir, 'gt')
            os.makedirs(train_ex_gt_dir, exist_ok=True)

            train_ex_good_rgb_dir = os.path.join(train_ex_rgb_dir, 'good')
            os.makedirs(train_ex_good_rgb_dir, exist_ok=True)
            train_ex_good_gt_dir = os.path.join(train_ex_gt_dir, 'good')
            os.makedirs(train_ex_good_gt_dir, exist_ok=True)

            train_good_rgb_dir = os.path.join(train_good_dir, 'rgb')
            images = os.listdir(train_good_rgb_dir)
            for image in images:
                org_img_dir = os.path.join(train_good_rgb_dir, image)
                new_img_dir = os.path.join(train_ex_good_rgb_dir, image)
                rgb_pil = Image.open(org_img_dir)
                org_h, org_w = rgb_pil.size
                mask_pil = Image.fromarray((np.zeros((org_h, org_w))).astype(np.uint8))
                mask_pil_dir = os.path.join(train_ex_good_gt_dir, image)
                rgb_pil.save(new_img_dir)
                mask_pil.save(mask_pil_dir)
            # ---------------------------------------------------------------------------------------- #
            validation_dir = os.path.join(cat_dir, 'validation')
            validation_good_dir = os.path.join(validation_dir, 'good')
            validation_good_rgb_dir = os.path.join(validation_good_dir, 'rgb')
            images = os.listdir(validation_good_rgb_dir)
            for image in images:
                name, ext = os.path.splitext(image)
                org_img_dir = os.path.join(validation_good_rgb_dir, image)
                new_img_dir = os.path.join(train_ex_good_rgb_dir, f'val_{name}{ext}')
                rgb_pil = Image.open(org_img_dir)
                org_h, org_w = rgb_pil.size
                mask_pil = Image.fromarray((np.zeros((org_h, org_w))).astype(np.uint8))
                mask_pil_dir = os.path.join(train_ex_good_gt_dir, f'val_{name}{ext}')
                rgb_pil.save(new_img_dir)
                mask_pil.save(mask_pil_dir)
            # ---------------------------------------------------------------------------------------- #
            test_dir = os.path.join(cat_dir, 'test')
            defets = os.listdir(test_dir)
            for defet in defets:
                org_defect_dir = os.path.join(test_dir, defet)
                org_defect_rgb_dir = os.path.join(org_defect_dir, 'rgb')
                org_defect_gt_dir = os.path.join(org_defect_dir, 'gt')

                train_defect_rgb_dir = os.path.join(train_ex_rgb_dir, defet)
                os.makedirs(train_defect_rgb_dir, exist_ok=True)
                train_defect_gt_dir = os.path.join(train_ex_gt_dir, defet)
                os.makedirs(train_defect_gt_dir, exist_ok=True)

                rgb_images = os.listdir(org_defect_rgb_dir)
                train_num = int(len(rgb_images) * 0.8)
                for i, image in enumerate(rgb_images) :
                    org_img_dir = os.path.join(org_defect_rgb_dir, image)
                    mask_img_dir = os.path.join(org_defect_gt_dir, image)
                    if i < train_num:
                        new_img_dir = os.path.join(train_defect_rgb_dir, image)
                        new_mask_dir = os.path.join(train_defect_gt_dir, image)

                        rgb_pil = Image.open(org_img_dir)
                        mask_pil = Image.open(mask_img_dir)
                        rgb_pil.save(new_img_dir)
                        mask_pil.save(new_mask_dir)








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='tire_org')
    args = parser.parse_args()
    main(args)