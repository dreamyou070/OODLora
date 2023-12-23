import os, shutil
from PIL import Image

folder = 'sample_data'
classes = os.listdir(folder)
for class_name in classes:  # bagel
    class_dir = os.path.join(folder, class_name)

    compare_folder = os.path.join(class_dir, 'compare')
    os.makedirs(compare_folder, exist_ok=True)

    bad_folder = os.path.join(class_dir, 'bad')
    corrected_folder = os.path.join(class_dir, 'corrected')
    gt_folder = os.path.join(class_dir, 'gt')
    cats = os.listdir(gt_folder)

    for cat in cats:

        compare_cat_dir = os.path.join(compare_folder, cat)
        os.makedirs(compare_cat_dir, exist_ok=True)

        bad_cat_dir = os.path.join(bad_folder, cat)
        corrected_cat_dir = os.path.join(corrected_folder, cat)
        gt_cat_dir = os.path.join(gt_folder, cat)

        images = os.listdir(gt_cat_dir)

        for image in images:
            name, ext = image.split('.')

            org_img_dir = os.path.join(bad_cat_dir, image)
            Image.open(org_img_dir).resize((512, 512)).save(os.path.join(compare_cat_dir, image))

            img_dir = os.path.join(corrected_cat_dir, image)
            Image.open(img_dir).resize((512, 512)).save(os.path.join(compare_cat_dir, f'{name}_inpaint.{ext}'))

            mask_img_dir = os.path.join(gt_cat_dir, image)
            Image.open(mask_img_dir).resize((512, 512)).save(os.path.join(compare_cat_dir, f'{name}_mask.{ext}'))