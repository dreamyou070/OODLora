#from transformers import ViTImageProcessor, ViTModel
#from PIL import Image
import os

def main() :

    base_folder = 'sample'
    lora_epochs = os.listdir(base_folder)
    for lora_epoch in lora_epochs:
        lora_epoch_folder = os.path.join(base_folder, lora_epoch)
        train_datasets = os.path.join(lora_epoch_folder, 'train_dataset')
        test_datasets = os.path.join(lora_epoch_folder, 'test_dataset')
        cats = os.listdir(train_datasets)
        for cat in cats:
            train_cat_dir = os.path.join(train_datasets, cat)
            images = os.listdir(train_cat_dir)
            for image in images:
                image_dir = os.path.join(train_cat_dir, image)
                if 'nois' in image :
                    os.remove(image_dir)
                if 'student' in image :
                    os.remove(image_dir)
                if 'recon_start' in image :
                    os.remove(image_dir)
                if 'recon_0' in image :
                    pure_name = image.split('_')[0]
                    pure_img_dir = os.path.join(train_cat_dir, f'{pure_name}.png')
                    mask_img_dir = os.path.join(train_cat_dir, f'{pure_name}_mask.png')
                    recon_img_dir = os.path.join(train_cat_dir, f'{pure_name}_recon_0.png')
                    new_folder = os.path.join(train_cat_dir, str(pure_name))
                    os.mkdir(new_folder)
                    new_pure_img_dir = os.path.join(new_folder, f'{pure_name}.png')
                    new_mask_img_dir = os.path.join(new_folder, f'{pure_name}_mask.png')
                    new_recon_img_dir = os.path.join(new_folder, f'{pure_name}_recon_0.png')
                    os.rename(pure_img_dir, new_pure_img_dir)
                    os.rename(mask_img_dir, new_mask_img_dir)
                    os.rename(recon_img_dir, new_recon_img_dir)
            #test_cat_dir = os.path.join(test_datasets, cat)

if __name__ == '__main__' :
    main()