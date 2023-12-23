import os, shutil
from PIL import Image

f_der = 'sample_data'
classes = os.listdir(f_der)
for class_name in classes:  # bagel
    if class_name == 'cookie' :

        class_dir = os.path.join(f_der, class_name)

        train_dir = os.path.join(class_dir, 'train')

        # ---------------------------------------------------------------
        compare_dir = os.path.join(train_dir, 'compare')

        bad_dir = os.path.join(train_dir, 'bad')
        os.makedirs(bad_dir, exist_ok=True)

        corrected_dir = os.path.join(train_dir, 'corrected')
        os.makedirs(corrected_dir, exist_ok=True)

        gt_dir = os.path.join(train_dir, 'gt')
        os.makedirs(gt_dir, exist_ok=True)

        cats = os.listdir(compare_dir)

        for cat in cats :

            cat_dir = os.path.join(compare_dir, cat)

            bad_cat_dir = os.path.join(bad_dir, cat)
            corrected_cat_dir = os.path.join(corrected_dir, cat)
            gt_cat_dir = os.path.join(gt_dir, cat)
            os.makedirs(bad_cat_dir, exist_ok=True)
            os.makedirs(corrected_cat_dir, exist_ok=True)
            os.makedirs(gt_cat_dir, exist_ok=True)

            images = os.listdir(cat_dir)

            for image in images :
                name, ext = image.split('.')

                if 'inpaint' in name :
                    img_dir = os.path.join(cat_dir, image)

                    num = name.split('_')[0]
                    new_img_dir = os.path.join(corrected_cat_dir, f'{num}.{ext}')
                    Image.open(img_dir).resize((512, 512)).save(new_img_dir)
                    os.remove(img_dir)

                elif 'mask' in name :
                    img_dir = os.path.join(cat_dir, image)
                    num = name.split('_')[0]
                    new_img_dir = os.path.join(gt_cat_dir, f'{num}.{ext}')
                    shutil.move(img_dir, new_img_dir)

                else :
                    img_dir = os.path.join(cat_dir, image)
                    new_img_dir = os.path.join(bad_cat_dir, image)
                    shutil.move(img_dir, new_img_dir)

        # remove compare dif
        shutil.rmtree(compare_dir)


        test_dir = os.path.join(class_dir, 'test')
        cats = os.listdir(test_dir)

        rgb_dir = os.path.join(test_dir, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        gt_dir = os.path.join(test_dir, 'gt')
        os.makedirs(gt_dir, exist_ok=True)

        for cat in cats :

            cat_dir = os.path.join(test_dir, cat)

            rgb_cat_dir = os.path.join(rgb_dir, cat)
            os.makedirs(rgb_cat_dir, exist_ok=True)
            gt_cat_dir = os.path.join(gt_dir, cat)
            os.makedirs(gt_cat_dir, exist_ok=True)

            images = os.listdir(cat_dir)

            for image in images :
                img_dir = os.path.join(cat_dir, image)
                name, ext = image.split('.')
                if 'mask' in image :
                    n_ = name.split('_')[0]
                    os.rename(img_dir, os.path.join(gt_cat_dir, f'{n_}.{ext}'))
                elif 'inpaint' in image :
                    os.remove(img_dir)
                else :
                    os.rename(img_dir, os.path.join(rgb_cat_dir, image))

        categories = os.listdir(test_dir)
        for category in categories :
            if category != 'rgb' and category != 'gt' :
                shutil.rmtree(os.path.join(test_dir, category))