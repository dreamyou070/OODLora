import os, shutil

f_der = 'sample_data'
classes = os.listdir(f_der)
for class_name in classes:  # bagel
    if class_name != 'bagel' and class_name != 'cable_gland' and class_name != 'carrot' and class_name != 'cookie':
        c_dir = os.path.join(f_der, class_name)
        folders = os.listdir(c_dir)
        print(folders)

        for folder in folders:
            before_dir = os.path.join(c_dir, folder)
            train_dir = os.path.join(c_dir, 'train')
            os.makedirs(train_dir, exist_ok=True)
            next_dir = os.path.join(train_dir, folder)
            shutil.move(before_dir, next_dir)

        os.makedirs(os.path.join(c_dir, 'test'), exist_ok=True)