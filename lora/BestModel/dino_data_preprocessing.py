#from transformers import ViTImageProcessor, ViTModel
#from PIL import Image
import os

def main() :

    base_folder = '../../result/MVTec3D-AD_experiment/potato/vae_training'
    experiments = os.listdir(base_folder)
    for experiment in experiments:
        if '1_' not in experiment:
            experiment_folder = os.path.join(base_folder, experiment)
            inference_dir = os.path.join(experiment_folder, 'inference_finding_best_epoch')
            student_epochs = os.listdir(base_folder)
            for student_epoch in student_epochs:
                student_epoch_folder = os.path.join(base_folder, student_epoch)
                training_dir = os.path.join(student_epoch_folder, 'training_dataset')
                test_dir = os.path.join(student_epoch_folder, 'test_dataset')

                cats = os.listdir(training_dir)
                for cat in cats:
                    cat_dir = os.path.join(training_dir, cat)
                    images = os.listdir(cat_dir)
                    for image in images:
                        name, ext = os.path.splitext(image)
                        if '_' in name :
                            name.split('_')[0]
                        image_dir = os.path.join(cat_dir, image)
                        new_folder = os.path.join(cat_dir, name)
                        os.makedirs(new_folder, exist_ok=True)
                        new_image_dir = os.path.join(new_folder, image)
                        os.rename(image_dir, new_image_dir)


                        if 'nois' in image :
                            os.remove(image_dir)
                        if 'student' in image :
                            os.remove(image_dir)
                        if 'recon_start' in image :
                            os.remove(image_dir)
                        if 'recon_0' in image :
                            pure_name = image.split('_')[0]
                            pure_img_dir = os.path.join(cat_dir, f'{pure_name}.png')
                            os.remove(pure_img_dir)
                            mask_img_dir = os.path.join(cat_dir, f'{pure_name}_mask.png')
                            os.remove(mask_img_dir)
                            recon_img_dir = os.path.join(cat_dir, f'{pure_name}_recon_0.png')
                            os.remove(recon_img_dir)



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