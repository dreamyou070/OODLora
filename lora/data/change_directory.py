from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import random

trg_ratio = 0.8

def main(args):


    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        if cat == args.trg_cat :
            cat_dir = os.path.join(base_folder, f'{cat}')

            train_dir = os.path.join(cat_dir, 'train')
            test_rgb_dir = os.path.join(cat_dir, 'test')
            test_gt_dir = os.path.join(cat_dir, 'ground_truth')


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

            categoriess = os.listdir(test_rgb_dir)
            for categor in categoriess:

                org_test_rgb_dir = os.path.join(test_rgb_dir, categor)
                org_test_gt_dir =  os.path.join(test_gt_dir, categor)

                new_test_rgb_dir = os.path.join(test_ex_rgb_dir, categor)
                new_test_gt_dir = os.path.join(test_ex_gt_dir, categor)
                os.makedirs(new_test_rgb_dir, exist_ok=True)
                os.makedirs(new_test_gt_dir, exist_ok=True)

                new_train_rgb_dir = os.path.join(train_ex_rgb_dir, categor)
                new_train_gt_dir = os.path.join(train_ex_gt_dir, categor)
                os.makedirs(new_train_rgb_dir, exist_ok=True)
                os.makedirs(new_train_gt_dir, exist_ok=True)

                if 'good' not in categor:
                    test_images = os.listdir(org_test_rgb_dir)
                    total_num = len(test_images)
                    test_num = int(total_num * 0.2)
                    for i, t_img in enumerate(test_images) :
                        name, ext = os.path.splitext(t_img)
                        org_rgb_dir = os.path.join(org_test_rgb_dir, t_img)
                        org_gt_dir = os.path.join(org_test_gt_dir, f'{name}_mask{ext}')

                        if i < test_num :
                            new_rgb_dir = os.path.join(new_test_rgb_dir, t_img)
                            new_gt_dir = os.path.join(new_test_gt_dir, t_img)
                        else :
                            new_rgb_dir = os.path.join(new_train_rgb_dir, t_img)
                            new_gt_dir = os.path.join(new_train_gt_dir, t_img)
                        Image.open(org_rgb_dir).resize((512,512)).save(new_rgb_dir)
                        gt = Image.open(org_gt_dir).resize((512,512)).convert('L')
                        gt.save(new_gt_dir)

                else :
                    for i, t_img in enumerate(test_images):
                        name, ext = os.path.splitext(t_img)
                        org_rgb_dir = os.path.join(org_test_rgb_dir, t_img)
                        #org_gt_dir = os.path.join(org_test_gt_dir, f'{name}_mask{ext}')
                        import numpy as np
                        mask_img = np.zeros((512, 512)).astype('uint8')
                        mask_img = Image.fromarray(mask_img)

                        new_rgb_dir = os.path.join(new_train_rgb_dir, t_img)
                        new_gt_dir = os.path.join(new_train_gt_dir, t_img)
                        Image.open(org_rgb_dir).resize((512, 512)).save(new_rgb_dir)
                        mask_img.save(new_gt_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'../../../../MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='bottle')
    args = parser.parse_args()
    main(args)