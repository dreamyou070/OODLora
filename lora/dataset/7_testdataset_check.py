import os, shutil
from PIL import Image

f_der = 'sample_data'
classes = os.listdir(f_der)
trg_num = 1000
for class_name in classes:  # bagel
    if class_name == 'carrot':
        class_dir = os.path.join(f_der, class_name)
        test_dir = os.path.join(class_dir, 'test')
        gt_folder = os.path.join(test_dir, 'gt')
        cats = os.listdir(gt_folder)
        for cat in cats:
            cat_folder = os.path.join(gt_folder, cat)
            images = os.listdir(cat_folder)
            for image in images:
                name, ext = os.path.splitext(image)
                pure_name = name.split('_')[0]
                org_img_dir = os.path.join(cat_folder, image)
                new_img_dir = os.path.join(cat_folder, f'{pure_name}{ext}')
                os.rename(org_img_dir, new_img_dir)

            """
            if 'rgb' not in cat and 'gt' not in cat:

                cat_folder = os.path.join(test_dir, cat)

                rgb_folder = os.path.join(test_dir, 'rgb')
                os.makedirs(rgb_folder, exist_ok=True)
                rgb_cat_folder = os.path.join(rgb_folder, cat)
                os.makedirs(rgb_cat_folder, exist_ok=True)

                gt_folder = os.path.join(test_dir, 'gt')
                os.makedirs(gt_folder, exist_ok=True)
                gt_cat_folder = os.path.join(gt_folder, cat)
                os.makedirs(gt_cat_folder, exist_ok=True)

                images = os.listdir(cat_folder)
                for image in images:
                    name, ext = os.path.splitext(image)
                    if 'mask' in name:
                        mask_img_dir = os.path.join(cat_folder, image)
                        #Image.open(mask_img_dir).resize((512, 512)).save(os.path.join(gt_cat_folder, image))
                        os.remove(mask_img_dir)
                    
                    elif 'inpaint' in name :
                        rgb_img_dir = os.path.join(cat_folder, image)
                        os.remove(rgb_img_dir)
                    elif '_' not in name :
                        rgb_img_dir = os.path.join(cat_folder, image)
                        Image.open(rgb_img_dir).resize((512, 512)).save(os.path.join(rgb_cat_folder, image))
            """

