import os, shutil
from PIL import Image
folder = 'sample_data'
classes = os.listdir(folder)
for class_name in classes:
    class_dir = os.path.join(folder, class_name)
    compare_folder = os.path.join(class_dir, 'compare')
    cats = os.listdir(compare_folder)
    train_folder = os.path.join(class_dir, 'train')
    bad_folder = os.path.join(train_folder, 'bad')
    corrected_folder = os.path.join(train_folder, 'corrected')
    gt_folder = os.path.join(train_folder, 'gt')
    for cat in cats:
        cat_dir = os.path.join(compare_folder, cat)

        bad_cat_dir = os.path.join(bad_folder, cat)
        corrected_cat_dir = os.path.join(corrected_folder, cat)
        gt_cat_dir = os.path.join(gt_folder, cat)

        images = os.listdir(cat_dir)
        for image in images:
            name, ext = image.split('.')
            if 'inpainted' in name:
                img_dir = os.path.join(cat_dir, image)
                number = name.split('_')[0]
                new_img_dir = os.path.join(corrected_cat_dir,f'{number}.{ext}')
                os.rename(img_dir, new_img_dir)

            elif 'mask' in name:
                img_dir = os.path.join(cat_dir, image)
                number = name.split('_')[0]
                new_img_dir = os.path.join(gt_cat_dir,f'{number}.{ext}')
                os.rename(img_dir, new_img_dir)

            else:
                img_dir = os.path.join(cat_dir, image)
                number = name.split('_')[0]
                new_img_dir = os.path.join(bad_cat_dir,f'{number}.{ext}')
                os.rename(img_dir, new_img_dir)

    test_folder = os.path.join(class_dir, 'test/rgb')
    test_gt_folder = os.path.join(class_dir, 'test/gt')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(test_gt_folder, exist_ok=True)
    cats = os.listdir(test_folder)
    for cat in cats:
        cat_dir  = os.path.join(test_folder, cat)
        gt_cat_dir = os.path.join(test_gt_folder, cat)
        os.makedirs(gt_cat_dir, exist_ok=True)
        images = os.listdir(cat_dir)
        for image in images:
            name, ext = image.split('.')
            img_dir = os.path.join(cat_dir, image)
            if 'mask' in name:
                number = name.split('_')[0]
                new_img_dir = os.path.join(gt_cat_dir,f'{number}.{ext}')
                os.rename(img_dir, new_img_dir)

            elif 'inpainted' in name:
                os.remove(img_dir)


"""
for cls in classes:
    class_train_folder = os.path.join(folder, cls, 'train')
    class_test_folder = os.path.join(folder, cls, 'test')
    os.makedirs(class_train_folder, exist_ok=True)
    os.makedirs(class_train_folder, exist_ok=True)

    bad_folder = os.path.join(folder, cls, 'bad')
    new_bad_folder = os.path.join(class_train_folder, 'bad')
    shutil.move(bad_folder, new_bad_folder)

    corrected_folder = os.path.join(folder, cls, 'corrected')
    new_corrected_folder = os.path.join(class_train_folder, 'corrected')
    shutil.move(corrected_folder, new_corrected_folder)

for cls in classes:
    cls_dir = os.path.join(folder, cls)
    compare_folder = os.path.join(cls_dir, 'compare/hole')
    images = os.listdir(compare_folder)
    for image in images:
        img_dir = os.path.join(compare_folder, image)
        img = Image.open(img_dir).resize((512,512))
        img.save(img_dir)
"""
"""        
train_folder = os.path.join(cls_dir, 'train')
bad_folder = os.path.join(train_folder, 'bad')
corrected_folder = os.path.join(train_folder, 'corrected')
gt_folder = os.path.join(train_folder, 'gt')

cats = os.listdir(bad_folder)
for cat in cats:
    if cat == 'hole' :
        compare_cat_dir = os.path.join(compare_folder, cat)
        os.makedirs(compare_cat_dir, exist_ok=True)
        bad_cat_dir = os.path.join(bad_folder, cat)
        corrected_cat_dir = os.path.join(corrected_folder, cat)
        gt_cat_dir = os.path.join(gt_folder, cat)

        images = os.listdir(bad_cat_dir)
        for image in images:
            name, ext = image.split('.')

            bad_img_dir = os.path.join(bad_cat_dir, image)
            new_img_dir = os.path.join(compare_cat_dir, f'{name}.{ext}')
            os.rename(bad_img_dir, new_img_dir)

            corrected_img_dir = os.path.join(corrected_cat_dir, image)
            new_img_dir = os.path.join(compare_cat_dir, f'{name}_inpainted.{ext}')
            os.rename(corrected_img_dir, new_img_dir)

            gt_img_dir = os.path.join(gt_cat_dir, image)
            new_img_dir = os.path.join(compare_cat_dir, f'{name}_mask.{ext}')
            os.rename(gt_img_dir, new_img_dir)
"""
