import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
from PIL import Image
import argparse
import shutil

def main(args):
    print(f'\n step 1. make model')
    device = args.device
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                          cache_dir=r'../../../../pretrained_stable_diffusion').to(device)

    """
    print(f'\n step 2. dataset')

    data_folder = args.data_folder
    classes = os.listdir(data_folder)

    for cls in classes:
        class_dir = os.path.join(data_folder, cls)

        # (1) train dir make
        train_dir = os.path.join(class_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)

        bad_dir = os.path.join(class_dir, 'bad')
        corrected_dir = os.path.join(class_dir, 'corrected')
        damages = os.listdir(corrected_dir)
        for damage in damages:
            if damage != 'good':

                train_anomaly_dir = os.path.join(train_dir, damage)
                os.makedirs(train_anomaly_dir, exist_ok=True)

                bad_damage_dir = os.path.join(bad_dir,       damage) # combined
                damage_dir     = os.path.join(corrected_dir, damage)
                train_damage_dir = os.path.join(train_dir, damage)
                os.makedirs(train_damage_dir, exist_ok=True)

                images = os.listdir(damage_dir)

                for image in images:

                    bad_image_path = os.path.join(bad_damage_dir, image)
                    image_path = os.path.join(damage_dir, image)

                    pil = Image.open(image_path)
                    np_img = np.array(pil)
                    np_sum = np.sum(np_img)

                    if np_sum == 0:

                        redir = os.path.join(train_damage_dir, image)
                        os.rename(bad_image_path, redir)

                        os.remove(image_path)
    """
    """
    classes = os.listdir(args.data_folder)
    parent, child = os.path.split(args.data_folder)
    save_folder = os.path.join(parent, f'{child}_Experiment')
    for cls in classes:
        if cls != 'carpet' :
            if cls != 'erase' :
                class_dir = os.path.join(args.data_folder, cls)
                save_class_dir = os.path.join(save_folder, cls)
                test_dir = os.path.join(class_dir, 'test')

                save_test_dir = os.path.join(save_class_dir, f'test/bad')
                test_mask_dir = os.path.join(save_class_dir, 'test_mask')
                os.makedirs(save_test_dir, exist_ok=True)

                train_mask_dir = os.path.join(save_class_dir, 'train_mask')
                os.makedirs(train_mask_dir, exist_ok=True)

                damages = os.listdir(test_dir)
                for damage in damages:
                    damage_dir   = os.path.join(test_dir, f'{damage}/gt')
                    org_images = os.listdir(damage_dir)

                    save_damage_dir = os.path.join(save_test_dir, damage)
                    tests = os.listdir(save_damage_dir)

                    test_damage_dir = os.path.join(test_mask_dir, damage)
                    os.makedirs(test_damage_dir, exist_ok=True)
                    train_damage_dir = os.path.join(train_mask_dir, damage)
                    os.makedirs(train_damage_dir, exist_ok=True)

                    for org_image in org_images:
                        if org_image in tests:
                            org_image_path = os.path.join(damage_dir, org_image)
                            re_image_path = os.path.join(test_damage_dir, org_image)
                            shutil.copy(org_image_path, re_image_path)
                        else :
                            org_image_path = os.path.join(damage_dir, org_image)
                            re_image_path = os.path.join(train_damage_dir, org_image)
                            shutil.copy(org_image_path, re_image_path)
    """
    """
    classes = os.listdir(args.data_folder)
    for cls in classes:
        cls_dir = os.path.join(args.data_folder, f'{cls}/train')
        damages = os.listdir(cls_dir)
        for damage in damages:
            damage_dir = os.path.join(args.data_folder, f'{cls}/train/{damage}')
            images = os.listdir(damage_dir)
            if len(images) == 0 :
                rgb_dir = os.path.join(args.data_folder, f'{cls}/test/bad/{damage}')
                rgb_corrected_dir = os.path.join(args.data_folder, f'{cls}/test/corrected/{damage}')
                rgb_images = os.listdir(rgb_dir)
                rgb_corrected_images = os.listdir(rgb_corrected_dir)
                total_num = int(len(rgb_images) * 0.2) + 1

                for i in range(total_num) :
                    org_img_dir = os.path.join(rgb_dir, rgb_images[i])
                    re_img_dir = os.path.join(args.data_folder, f'{cls}/train/{damage}/{rgb_images[i]}')
                    os.rename(org_img_dir, re_img_dir)

                    org_mask_dir = os.path.join(args.data_folder, f'{cls}/test_mask/{damage}/{rgb_images[i]}')
                    re_mask_dir = os.path.join(args.data_folder, f'{cls}/train_mask/{damage}/{rgb_images[i]}')
                    os.rename(org_mask_dir, re_mask_dir)

                    org_corrected_dir = os.path.join(rgb_corrected_dir, rgb_corrected_images[i])
                    os.remove(org_corrected_dir)
    """
    classes = os.listdir(args.data_folder)
    for cls in classes:
        cls_dir = os.path.join(args.data_folder, f'{cls}/test')
        bad_dir = os.path.join(cls_dir, 'bad')
        corrected_dir = os.path.join(cls_dir, 'corrected')
        category = os.listdir(bad_dir)
        for cate in category:
            if cate != 'good' :
                org_cat_dir = os.path.join(bad_dir, cate)
                re_cat_dir = os.path.join(bad_dir, f'30_{cate}')
                shutil.move(org_cat_dir, re_cat_dir)

                org_cat_dir = os.path.join(corrected_dir, cate)
                re_cat_dir = os.path.join(corrected_dir, f'30_{cate}')
                shutil.move(org_cat_dir, re_cat_dir)
            else :
                org_cat_dir = os.path.join(bad_dir, cate)
                re_cat_dir = os.path.join(bad_dir, f'50_{cate}')
                shutil.move(org_cat_dir, re_cat_dir)

                org_cat_dir = os.path.join(corrected_dir, cate)
                re_cat_dir = os.path.join(corrected_dir, f'50_{cate}')
                shutil.move(org_cat_dir, re_cat_dir)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='../../../../MyData/anomaly_detection/MVTec3D-AD_Experiment')
    args = parser.parse_args()
    main(args)