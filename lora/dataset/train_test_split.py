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
    parent, child = os.path.split(args.data_folder)
    save_folder = os.path.join(parent, f'{child}_Experiment')
    os.makedirs(save_folder, exist_ok=True)

    data_folder = args.data_folder
    classes = os.listdir(data_folder)

    for cls in classes:
        class_dir = os.path.join(data_folder, cls)
        train_dir = os.path.join(class_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)

        corrected_dir = os.path.join(class_dir, 'corrected')
        bad_dir = os.path.join(class_dir, 'bad')
        damages = os.listdir(corrected_dir)
        for damage in damages:
            if damage != 'good':

                train_anomaly_dir = os.path.join(train_dir, damage)
                os.makedirs(train_anomaly_dir, exist_ok=True)

                damage_dir = os.path.join(corrected_dir, damage)
                images = os.listdir(damage_dir)
                for image in images:
                    image_path = os.path.join(damage_dir, image)
                    pil = Image.open(image_path)
                    np_img = np.array(pil)
                    np_sum = np.sum(np_img)
                    if np_sum == 0:
                        # save again
                        bad_dir = os.path.join(bad_dir, f'{damage}/{image}')
                        os.rename(bad_dir, os.path.join(train_anomaly_dir, image))

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