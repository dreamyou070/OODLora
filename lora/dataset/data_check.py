from diffusers import StableDiffusionInpaintPipeline
import torch, os
from PIL import Image
import argparse


def main(args):
    print(f'\n step 1. make model')
    device = args.device
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                          cache_dir=r'../../../../pretrained_stable_diffusion').to(device)

    print(f'\n step 2. dataset')
    parent, _ = os.path.split(args.data_folder)
    save_folder = os.path.join(parent, 'MVTec_Experiment')
    os.makedirs(save_folder, exist_ok=True)

    data_folder = args.data_folder
    classes = os.listdir(data_folder)
    before_classes = ['bottle','cable','capsule','carpet']
    for cls in classes:
        if cls not in before_classes:
            class_dir = os.path.join(data_folder, cls)
            save_class_dir = os.path.join(save_folder, cls)
            os.makedirs(save_class_dir, exist_ok=True)

            original_img_save_folder = os.path.join(save_class_dir, 'bad')
            os.makedirs(original_img_save_folder, exist_ok=True)
            inpaint_img_save_folder = os.path.join(save_class_dir, 'corrected')
            os.makedirs(inpaint_img_save_folder, exist_ok=True)

            prompt = cls
            test_folder = os.path.join(class_dir, 'test')
            ground_truth_folder = os.path.join(class_dir, 'ground_truth')
            categories = os.listdir(test_folder)
            for category in categories:
                if category != 'good':
                    test_cat         = os.path.join(test_folder, category)
                    ground_truth_cat = os.path.join(ground_truth_folder, category)

                    original_categori_dir = os.path.join(original_img_save_folder, category)
                    os.makedirs(original_categori_dir, exist_ok=True)
                    inpaint_categori_dir = os.path.join(inpaint_img_save_folder, category)
                    os.makedirs(inpaint_categori_dir, exist_ok=True)

                    images = os.listdir(test_cat)
                    for name_ in images:
                        name, ext = os.path.splitext(name_)
                        image_path = os.path.join(test_cat, name_)
                        mask_path = os.path.join(ground_truth_cat, f'{name}_mask{ext}')
                        image = pipe(prompt=prompt,
                                     image=Image.open(image_path).convert('RGB'),
                                     mask_image=Image.open(mask_path).convert('L'), ).images[0]
                        image.save(os.path.join(inpaint_categori_dir, name_))
                        original_image = Image.open(image_path)
                        original_image.save(os.path.join(original_categori_dir, name_))

            train_folder = os.path.join(class_dir, 'train', 'good')
            train_images = os.listdir(train_folder)
            original_save_folder = os.path.join(original_img_save_folder, 'good')
            copy_save_folder = os.path.join(inpaint_img_save_folder, 'good')
            os.makedirs(original_save_folder, exist_ok=True)
            os.makedirs(copy_save_folder, exist_ok=True)
            for i in train_images:
                image_path = os.path.join(train_folder, i)
                image = Image.open(image_path)
                image.save(os.path.join(original_save_folder, i))
                image.save(os.path.join(copy_save_folder, i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='../../../../MyData/anomaly_detection/MVTec')
    args = parser.parse_args()
    main(args)