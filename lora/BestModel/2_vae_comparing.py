# from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import os
import numpy as np


def main():
    base_folder = '../../result/MVTec3D-AD_experiment/potato/vae_training'
    save_folder = '../../result/MVTec3D-AD_experiment/potato/vae_training_scoring'
    os.makedirs(save_folder, exist_ok=True)
    experiments = os.listdir(base_folder)
    for experiment in experiments:

        if '1_' not in experiment  :
            experiment_folder = os.path.join(base_folder, experiment)

            inference_dir = os.path.join(experiment_folder, 'inference_finding_best_epoch')
            experiments_records = []
            student_epochs = os.listdir(inference_dir)
            for student_epoch in student_epochs:
                student_epoch_folder = os.path.join(inference_dir, student_epoch)
                training_dir = os.path.join(student_epoch_folder, 'training_dataset')
                test_dir = os.path.join(student_epoch_folder, 'test_dataset')
                training_score = 0
                cats = os.listdir(training_dir) ####################
                for cat in cats:
                    cat_dir = os.path.join(training_dir, cat)
                    numbers = os.listdir(cat_dir)
                    for number in numbers:
                        number_dir = os.path.join(cat_dir, number)
                        images = os.listdir(number_dir)
                        for image in images:
                            name, ext = os.path.splitext(image)
                            if '_' not in name :
                                base_img = os.path.join(number_dir, image)
                            elif 'recon' in name:
                                recon_img = os.path.join(number_dir, image)
                            else :
                                mask_img = os.path.join(number_dir, image)

                        x = np.array(Image.open(base_img))
                        x_hat = np.array(Image.open(recon_img))
                        mse = np.square(x - x_hat) ** 0.5
                        binary = np.where(mse > 0.5, 1, 0)

                        gt = np.array(Image.open(mask_img).to("RGB"))
                        gt = np.where(gt > 100, 1, 0)

                        error = np.square(binary - gt) ** 0.5
                        training_score += error


                test_score = 0
                cats = os.listdir(test_dir)  ####################
                for cat in cats:
                    cat_dir = os.path.join(test_dir, cat)
                    numbers = os.listdir(cat_dir)
                    for number in numbers:
                        number_dir = os.path.join(cat_dir, number)
                        images = os.listdir(number_dir)
                        for image in images:
                            name, ext = os.path.splitext(image)
                            if '_' not in name:
                                base_img = os.path.join(number_dir, image)
                            elif 'recon' in name:
                                recon_img = os.path.join(number_dir, image)
                            else:
                                mask_img = os.path.join(number_dir, image)

                        x = np.array(Image.open(base_img))
                        x_hat = np.array(Image.open(recon_img))
                        mse = np.square(x - x_hat) ** 0.5
                        binary = np.where(mse > 0.5, 1, 0)

                        gt = np.array(Image.open(mask_img).to("RGB"))
                        gt = np.where(gt > 100, 1, 0)

                        error = np.square(binary - gt) ** 0.5
                        test_score += error

                elem = [student_epoch, training_score, test_score]
                experiments_records.append(elem)
            experiments_record = os.path.join(save_folder, f'score_{experiment}.txt')
            with open(experiments_record, 'w') as f:
                f.write(f'epoch | training_score | test_score \n')
                for record in experiments_records:
                    f.write(f'{record[0]} | {record[1]} | {record[2]}\n')






if __name__ == '__main__':
    main()