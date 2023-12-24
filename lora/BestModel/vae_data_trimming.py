# from transformers import ViTImageProcessor, ViTModel
# from PIL import Image
import os


def main():
    base_folder = '../../result/MVTec3D-AD_experiment/potato/vae_training'
    experiments = os.listdir(base_folder)
    for experiment in experiments:
        if '3_' in experiment or '4_' in experiment :
            experiment_folder = os.path.join(base_folder, experiment)
            inference_dir = os.path.join(experiment_folder, 'inference_finding_best_epoch')
            student_epochs = os.listdir(inference_dir)
            for student_epoch in student_epochs:
                student_epoch_folder = os.path.join(inference_dir, student_epoch)
                training_dir = os.path.join(student_epoch_folder, 'training_dataset')
                test_dir = os.path.join(student_epoch_folder, 'test_dataset')

                cats = os.listdir(training_dir) ####################
                for cat in cats:
                    cat_dir = os.path.join(training_dir, cat)
                    sub_folders = os.listdir(cat_dir)
                    for sub_folder in sub_folders:
                        if '_' in sub_folder:
                            name = sub_folder.split('_')[0]
                            name_dir = os.path.join(cat_dir, name)
                            sub_folder_dir = os.path.join(cat_dir, sub_folder)
                            image = os.path.join(sub_folder_dir, f'{sub_folder}.png')
                            new_img_dir = os.path.join(name_dir, f'{sub_folder}.png')
                            os.rename(image, new_img_dir)

                cats = os.listdir(test_dir)  ####################
                for cat in cats:
                    cat_dir = os.path.join(test_dir, cat)
                    sub_folders = os.listdir(cat_dir)
                    for sub_folder in sub_folders:
                        if '_' in sub_folder:
                            #name = sub_folder.split('_')[0]
                            #name_dir = os.path.join(cat_dir, name)
                            sub_folder_dir = os.path.join(cat_dir, sub_folder)
                            os.rmdir(sub_folder_dir)
                            #image = os.path.join(sub_folder_dir, f'{sub_folder}.png')
                            #new_img_dir = os.path.join(name_dir, f'{sub_folder}.png')
                            #os.rename(image, new_img_dir)

                    """        
                    images = os.listdir(cat_dir)
                    for image in images:
                        name, ext = os.path.splitext(image)
                        if '_' in name:
                            name.split('_')[0]
                        image_dir = os.path.join(cat_dir, image)
                        new_folder = os.path.join(cat_dir, name)
                        os.makedirs(new_folder, exist_ok=True)
                        new_image_dir = os.path.join(new_folder, image)
                        os.rename(image_dir, new_image_dir)
                    """
                """
                cats = os.listdir(test_dir)
                for cat in cats:
                    cat_dir = os.path.join(test_dir, cat)
                    images = os.listdir(cat_dir)
                    for image in images:
                        name, ext = os.path.splitext(image)
                        if '_' in name:
                            name.split('_')[0]
                        image_dir = os.path.join(cat_dir, image)
                        new_folder = os.path.join(cat_dir, name)
                        os.makedirs(new_folder, exist_ok=True)
                        new_image_dir = os.path.join(new_folder, image)
                        os.rename(image_dir, new_image_dir)
                """

if __name__ == '__main__':
    main()