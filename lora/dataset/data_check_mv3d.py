import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image
import argparse


def main(args):
    print(f'\n step 1. make model')
    device = args.device

    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                     torch_dtype=torch.float16,
                                                     variant = "fp16",
                                                     cache_dir=r'../../../../pretrained_stable_diffusion').to(device)

    base_mask_dir = '../../../../MyData/anomaly_detection/base.png'

    """
    pipe.enable_model_cpu_offload() --> 
    """
    #
    #pipe.enable_xformers_memory_efficient_attention()


    print(f'\n step 2. dataset')
    parent, child = os.path.split(args.data_folder)
    save_folder = os.path.join(parent, f'{child}_Experiment_Recheck_SDXL')
    os.makedirs(save_folder, exist_ok=True)

    data_folder = args.data_folder
    classes = os.listdir(data_folder)
    for cls in classes:
        if cls == 'cookies' :
            print(f'cls : {cls}')
            class_dir = os.path.join(data_folder, cls)

            save_class_dir = os.path.join(save_folder, cls)
            os.makedirs(save_class_dir, exist_ok=True)

            original_img_save_folder = os.path.join(save_class_dir, 'bad')
            os.makedirs(original_img_save_folder, exist_ok=True)
            original_img_save_folder = os.path.join(save_class_dir, 'bad_model')
            os.makedirs(original_img_save_folder, exist_ok=True)
            inpaint_img_save_folder = os.path.join(save_class_dir, 'corrected')
            os.makedirs(inpaint_img_save_folder, exist_ok=True)

            prompt_list = '_'.split(cls)
            prompt = 'a circle black chocolate cookie' # + ' '.join(prompt_list)

            test_folder = os.path.join(class_dir, 'test')

            categories = os.listdir(test_folder)
            negative_prompt = ', '.join(prompt_list)

            for category in categories:
                if category != 'good':
                    test_cat = os.path.join(test_folder, category)
                    rgb_folder = os.path.join(test_cat, 'rgb')
                    gt_folder = os.path.join(test_cat, 'gt')
                    #ground_truth_cat = os.path.join(ground_truth_folder, category)

                    original_categori_dir = os.path.join(original_img_save_folder, category)
                    os.makedirs(original_categori_dir, exist_ok=True)
                    inpaint_categori_dir = os.path.join(inpaint_img_save_folder, category)
                    os.makedirs(inpaint_categori_dir, exist_ok=True)

                    images = os.listdir(rgb_folder)
                    for name_ in images:
                        name, ext = os.path.splitext(name_)
                        image_path = os.path.join(rgb_folder, name_)
                        pil = Image.open(image_path)
                        width, height = pil.size
                        print(f'original size : {width} , {height}')
                        mask_path = os.path.join(gt_folder, name_)
                        """
                        image = pipe(prompt=prompt,
                                     #negative_prompt = negative_prompt,
                                     image=Image.open(image_path).convert('RGB'),
                                     mask_image=Image.open(mask_path).convert('L'), ).images[0]
                        """
                        org_image = load_image(image_path).resize((1024, 1024))
                        mask_image = load_image(mask_path).resize((1024, 1024))
                        generator = torch.Generator(device="cuda").manual_seed(0)

                        image = pipe(
                            prompt=prompt,
                            image=org_image,
                            mask_image=mask_image,
                            guidance_scale=8.0,
                            num_inference_steps=20,  # steps between 15 and 30 work well for us
                            strength=0.99,  # make sure to use `strength` below 1.0
                            generator=generator,
                        ).images[0]
                        image = image.resize((width, height))
                        image.save(os.path.join(inpaint_categori_dir, name_))

                        original_image = Image.open(image_path).resize((512,512))
                        original_image.save(os.path.join(original_categori_dir, name_))

                        new_mask = load_image(base_mask_dir).resize((1024, 1024))
                        image = pipe(prompt=prompt,
                                     image=org_image,
                                     mask_image=new_mask,
                            guidance_scale=8.0,
                            num_inference_steps=20,  # steps between 15 and 30 work well for us
                            strength=0.99,  # make sure to use `strength` below 1.0
                            generator=generator,
                        ).images[0]
                        image = image.resize((width, height))
                        image.save(os.path.join(inpaint_categori_dir, name_))






            """
            train_folder = os.path.join(class_dir, 'train/good/rgb')
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
            """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='../../../../MyData/anomaly_detection/MVTec3D-AD_Recheck')
    args = parser.parse_args()
    main(args)