import os, shutil

f_der = 'sample_data'
classes = os.listdir(f_der)
for class_name in classes:  # bagel
    if class_name != 'bagel' :

        class_dir = os.path.join(f_der, class_name)
        train_dir = os.path.join(class_dir, 'train')
        test_dir = os.path.join(class_dir, 'test')
        """
        bad_dir = os.path.join(train_dir, 'bad')
        corrected_dir = os.path.join(train_dir, 'corrected')
        gt_dir = os.path.join(train_dir, 'gt')
        shutil.rmtree(bad_dir)
        shutil.rmtree(corrected_dir)
        shutil.rmtree(gt_dir)
        """
        compare_dir = os.path.join(train_dir, 'compare')
        cats = os.listdir(compare_dir)
        for cat in cats :
            os.makedirs(os.path.join(test_dir, cat), exist_ok=True)


