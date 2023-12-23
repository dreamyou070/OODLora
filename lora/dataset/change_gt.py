import os, shutil

gt_base_folder = '../../../../MyData/anomaly_detection/MVTec3D-AD'
save_base_folder = '../../../../MyData/anomaly_detection/MVTec3D-AD_Experiment_SDXL'
cats = os.listdir(gt_base_folder)
print(cats)
bad = ['carpet','erase','potato']
for cat in cats :
    if cat in bad :
        cat_dir = os.path.join(gt_base_folder, f'{cat}/test')
        save_cat_dir = os.path.join(save_base_folder, f'{cat}')
        categories = os.listdir(cat_dir)
        for category in categories :
            category_dir = os.path.join(cat_dir, category)
            gt_dir = os.path.join(category_dir, f'{category}/gt')
            images = os.listdir(gt_dir)
            for image in images :
                org_img_dir = os.path.join(category_dir, f'{category}/rgb/{image}')
                save_category_dir = os.path.join(save_cat_dir, f'gt/{category}')
                os.makedirs(save_category_dir, exist_ok=True)
                new_img_dir = os.path.join(save_category_dir, image)
                shutil.copy(org_img_dir, new_img_dir)





