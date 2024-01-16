import os

base_folder = r'../result/MVTec3D-AD_experiment/cookie/lora_training/anormal'
folders = os.listdir(base_folder)
for folder in folders:
    if folder == '1_1_res_64_up_16_up_cls_training_from_42_epoch' :
        folder_dir = os.path.join(base_folder, folder)
        files = os.listdir(folder_dir)
        for file in files :
            #if file == 'res_64_up_16_up' :
            if 'length_2' in file :
                file_dir_ = os.path.join(folder_dir,f'{file}' )
                f_dir = os.path.join(file_dir_,'trg_res_check_cross_attention_map')
                f_dir_ = os.path.join(f_dir, 'test_set_rearrange')
                cats = os.listdir(f_dir_)
                for cat in cats :
                    cat_dir = os.path.join(f_dir_, cat)
                    sub_cats = os.listdir(cat_dir)
                    for sub_cat in sub_cats :
                        sub_cat_dir = os.path.join(cat_dir, sub_cat)
                        images = os.listdir(sub_cat_dir)
                        for img in images :
                            img_dir = os.path.join(sub_cat_dir, img)
                            if 'rex_16' in img :
                                os.remove(img_dir)
                            else :
                                if 'cls' in img :
                                    os.remove(img_dir)
