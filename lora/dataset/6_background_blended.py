import os, shutil
from PIL import Image
import numpy as np
import argparse


def main(args) :

    base_folder = args.base_folder
    classes = os.listdir(base_folder)
    trg_num = 1000
    for class_name in classes:  # bagel

        if class_name == 'bagel':
            class_dir = os.path.join(base_folder, class_name)

            train_dir = os.path.join(class_dir, 'train')
            bad_train_dir = os.path.join(train_dir, 'bad')
            corrected_train_dir = os.path.join(train_dir, 'corrected')
            cor_2_dir = os.path.join(train_dir, 'cor_2')
            os.makedirs(cor_2_dir, exist_ok=True)
            gt_train_dir = os.path.join(train_dir, 'gt')
            test_dir = os.path.join(class_dir, 'test')
            categories = os.listdir(bad_train_dir)
            for category in categories:

                category_dir = os.path.join(bad_train_dir, category)
                category_corrected_dir = os.path.join(corrected_train_dir, category)
                category_cor_2_dir = os.path.join(cor_2_dir, category)
                os.makedirs(category_cor_2_dir, exist_ok=True)

                if '_' in category:
                    category = '_'.join(category.split('_')[1:])

                category_gt_dir = os.path.join(gt_train_dir, category)

                images = os.listdir(category_dir)

                for img in images:
                    background_img_dir = os.path.join(category_dir, img)
                    object_img_dir = os.path.join(category_corrected_dir, img)
                    mask_img_dir = os.path.join(category_gt_dir, img)

                    back_np = np.array(Image.open(background_img_dir))
                    obj_np = np.array(Image.open(object_img_dir))
                    mask_np = np.array(Image.open(mask_img_dir).convert('RGB'))
                    mask_np = np.where(mask_np > 100, 1, 0)
                    mask_np = mask_np.astype(np.uint8)
                    #Image.fromarray(mask_np*255).show()
                    new_np = back_np * (1 - mask_np) + obj_np * mask_np
                    new_img = Image.fromarray(new_np, "RGB")
                    new_img.save(os.path.join(category_cor_2_dir, img))

                """
                else :
                    category_corrected_dir = os.path.join(corrected_train_dir, category)
                    category_cor_2_dir = os.path.join(cor_2_dir, category)
                    os.makedirs(category_cor_2_dir, exist_ok=True)
                    shutil.copytree(category_corrected_dir, category_cor_2_dir)
                """

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default='../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL')
    main()