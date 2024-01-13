import os

base_folder = r'../result/MVTec3D-AD_experiment/cookie/lora_training/anormal'
folders = os.listdir(base_folder)
for folder in folders:

    folder_dir = os.path.join(base_folder, folder)
    files = os.listdir(folder_dir)
    for file in files :
        if 'length_2' in file :
            file_dir_ = os.path.join(folder_dir,f'{file}' )
            f_dir = os.path.join(file_dir_,'trg_res_check_cross_attention_map')
            f_dir_ = os.path.join(f_dir, 'test_set')
            save_folder = os.path.join(f_dir, 'test_set_rearrange')
            os.makedirs(save_folder, exist_ok = True)
            loras = os.listdir(f_dir_)
            for lora in loras :
                lora_dir = os.path.join(f_dir_, lora)
                cats = os.listdir(lora_dir)
                for cat in cats :
                    save_cat = os.path.join(save_folder, cat)
                    os.makedirs(save_cat, exist_ok=True)
                    cat_dir = os.path.join(lora_dir, cat)
                    sub_cats = os.listdir(cat_dir)
                    for sub_cat in sub_cats :

                        save_sub_cat = os.path.join(save_cat, sub_cat)
                        os.makedirs(save_sub_cat, exist_ok=True)
                        sub_cat_dir = os.path.join(cat_dir, sub_cat)
                        images = os.listdir(sub_cat_dir)
                        for img in images :
                            img_dir = os.path.join(sub_cat_dir, img)
                            re_img_dir = os.path.join(save_sub_cat, f'{lora}_{img}')
                            os.rename(img_dir,re_img_dir )
