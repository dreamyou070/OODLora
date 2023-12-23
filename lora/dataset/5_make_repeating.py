import os, shutil
from PIL import Image

f_der = 'sample_data'
classes = os.listdir(f_der)
trg_num = 1000
for class_name in classes:  # bagel
    if class_name == 'cookie':

        class_dir = os.path.join(f_der, class_name)
        train_dir = os.path.join(class_dir, 'train')
        test_dir = os.path.join(class_dir, 'test')
        bad_train_dir = os.path.join(train_dir, 'bad')
        corrected_train_dir = os.path.join(train_dir, 'corrected')
        categories = os.listdir(bad_train_dir)
        for category in categories:
            category_dir = os.path.join(bad_train_dir, category)
            category_corrected_dir = os.path.join(corrected_train_dir, category)
            images = os.listdir(category_dir)
            num = int(trg_num/len(images))
            new_names = os.path.join(bad_train_dir, f'{num}_{category}')
            new_names_corrected = os.path.join(corrected_train_dir, f'{num}_{category}')
            os.rename(category_dir, new_names)
            os.rename(category_corrected_dir, new_names_corrected)