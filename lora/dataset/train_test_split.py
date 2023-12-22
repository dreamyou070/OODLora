import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
from PIL import Image
import argparse


def main(args):
    print(f'\n step 1. make model')
    device = args.device
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                          cache_dir=r'../../../../pretrained_stable_diffusion').to(device)

    print(f'\n step 2. dataset')

    data_folder = args.data_folder
    classes = os.listdir(data_folder)

    for cls in classes:
        class_dir = os.path.join(data_folder, cls)

        # (1) train dir make
        train_dir = os.path.join(class_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)

        bad_dir = os.path.join(class_dir, 'bad')
        corrected_dir = os.path.join(class_dir, 'corrected')
        damages = os.listdir(corrected_dir)
        for damage in damages:
            if damage != 'good':

                train_anomaly_dir = os.path.join(train_dir, damage)
                os.makedirs(train_anomaly_dir, exist_ok=True)

                bad_damage_dir = os.path.join(bad_dir,       damage) # combined
                damage_dir     = os.path.join(corrected_dir, damage)
                train_damage_dir = os.path.join(train_dir, damage)
                os.makedirs(train_damage_dir, exist_ok=True)

                images = os.listdir(damage_dir)

                for image in images:

                    bad_image_path = os.path.join(bad_damage_dir, image)
                    image_path = os.path.join(damage_dir, image)

                    pil = Image.open(image_path)
                    np_img = np.array(pil)
                    np_sum = np.sum(np_img)

                    if np_sum == 0:

                        redir = os.path.join(train_damage_dir, image)
                        os.rename(bad_image_path, redir)

                        os.remove(image_path)
            """
            else :
                train_good_dir = os.path.join(train_dir, 'good')
                os.makedirs(train_good_dir, exist_ok=True)

                damage_dir = os.path.join(corrected_dir, damage)
                damage_corrected_dir = os.path.join(corrected_dir, 'good')
                images = os.listdir(damage_dir)
                for image in images:
                    image_path = os.path.join(damage_dir, image)
                    os.rename(image_path, os.path.join(train_good_dir, image))
                    a = os.path.join(damage_corrected_dir, image)
                    os.remove(a)
            """



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='../../../../MyData/anomaly_detection/MVTec3D-AD_Experiment')
    args = parser.parse_args()
    main(args)