import os
import argparse

def main(args) :

    base_folder = r'../result/MVTec3D-AD_experiment/foam/lora_training/anormal'
    folders = os.listdir(base_folder)

    for folder in folders:
        if folder == args.trg_folder :
            folder_dir = os.path.join(base_folder, folder)
            files = os.listdir(folder_dir)
            for file in files :

                if args.trg_word in file :
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trg_word', type=str, default='inference_truncate_length_2')
    parser.add_argument('--trg_folder', type=str, default='check_res_64_up_down_res_16_up_text_3_no_background_loss_recode')
    args = parser.parse_args()
    main(args)
