import os

base_folder = 'train_ex_2'

rgb_folder = os.path.join(base_folder, 'rgb')
gt_folder_ = os.path.join(base_folder, 'gt')
folders = os.listdir(rgb_folder)
for folder in folders :
    if folder == '30_good' :
        folder_dir = os.path.join(rgb_folder, '30_good')
        images = os.listdir(folder_dir)

        gt_folder = os.path.join(gt_folder_, '10_good')

        new_gt_folder = os.path.join(gt_folder_, '30_good')


        for image in images :
            org_gt_dir = os.path.join(gt_folder, image)
            new_gt_dir = os.path.join(new_gt_folder, f'{image}')
            os.rename(org_gt_dir, new_gt_dir)