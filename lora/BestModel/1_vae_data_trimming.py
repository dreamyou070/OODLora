# from transformers import ViTImageProcessor, ViTModel
# from PIL import Image
import os


def main():
    base_folder = '../../result/MVTec3D-AD_experiment/potato/vae_training'
    experiments = os.listdir(base_folder)
    for experiment in experiments:
        if '2_' in experiment  :
            experiment_folder = os.path.join(base_folder, experiment)
            inference_dir = os.path.join(experiment_folder, 'inference_finding_best_epoch')
            student_epochs = os.listdir(inference_dir)
            for student_epoch in student_epochs:
                student_epoch_folder = os.path.join(inference_dir, student_epoch)
                training_dir = os.path.join(student_epoch_folder, 'training_dataset')
                test_dir = os.path.join(student_epoch_folder, 'test_dataset')

                cats = os.listdir(training_dir)  ####################
                for cat in cats:
                    cat_dir = os.path.join(training_dir, cat)
                    folders = os.listdir(cat_dir)
                    for folder in folders:
                        if '_' in folder:

                            folder_dir = os.path.join(cat_dir, folder)
                            os.rmdir(folder_dir)

                cats = os.listdir(test_dir)  ####################
                for cat in cats:
                    cat_dir = os.path.join(test_dir, cat)
                    folders = os.listdir(cat_dir)
                    for folder in folders :
                        if '_' in folder :
                            folder_dir = os.path.join(cat_dir, folder)
                            os.rmdir(folder_dir)


if __name__ == '__main__':
    main()