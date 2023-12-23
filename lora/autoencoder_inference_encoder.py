import argparse
from diffusers.models.vae import DiagonalGaussianDistribution
import os
import random
from accelerate.utils import set_seed
import library.train_util as train_util
import library.config_util as config_util
import torch
from utils.image_utils import load_image, IMAGE_TRANSFORMS
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from STTraining import Encoder_Teacher, Encoder_Student
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None
from accelerate import Accelerator

def main(args):

    print(f'\n step 1. setting')
    print(f' (1) session')
    if args.process_title:
        setproctitle(args.process_title)
    else:
        setproctitle('parksooyeon')
    session_id = random.randint(0, 2 ** 32)

    print(f' (2) seed')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f' (3) accelerator')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,)

    print(f'\n step 2. model')
    print(f' (1) mixed precision and model')
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32
    print(f' (2) model')
    text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

    print(f' (3) making encoder of vae')
    vae_encoder = vae.encoder
    vae_encoder_quantize = vae.quant_conv
    vae_encoder.requires_grad_(False)
    vae_encoder.eval()
    vae_encoder.to(accelerator.device, dtype=vae_dtype)
    vae_encoder_quantize.requires_grad_(False)
    vae_encoder_quantize.eval()
    vae_encoder_quantize.to(accelerator.device, dtype=vae_dtype)
    vae.to(accelerator.device, dtype=vae_dtype)

    config_dict = vae.config


    student_vae = AutoencoderKL.from_config(config_dict)
    student_vae_encoder = student_vae.encoder
    student_vae_encoder_quantize = student_vae.quant_conv
    student = Encoder_Student(student_vae_encoder, student_vae_encoder_quantize)


    print(f' (4) making decoder of vae')
    student_pretrained_dir = args.student_pretrained_dir
    model_state_dict = torch.load(student_pretrained_dir, map_location="cpu")
    state_dict = {}
    for k, v in model_state_dict.items():
        k_ = '.'.join(k.split('.')[1:])
        state_dict[k_] = v
    student.load_state_dict(state_dict, strict=True)
    student.requires_grad_(False)
    student.eval()
    student.to(accelerator.device, dtype=vae_dtype)

    student_epoch = os.path.split(student_pretrained_dir)[-1]
    student_epoch = os.path.splitext(student_epoch)[0]
    student_epoch = int(student_epoch.split('_')[-1])
    print(f'student_epoch: {student_epoch}')

    def recon(mask_dir, sample_data_dir, mask_save_dir, save_dir, compare_save_dir):
        pil_img = Image.open(sample_data_dir)
        h, w = args.resolution.split(',')[0], args.resolution.split(',')[1]

        if mask_dir is not None :
            mask_pil_img = Image.open(mask_dir)
            mask_pil_img = mask_pil_img.resize((int(h.strip()), int(w.strip())), Image.BICUBIC)
            mask_pil_img.save(mask_save_dir)

        pil_img = pil_img.resize((int(h.strip()), int(w.strip())), Image.BICUBIC)
        pil_img.save(compare_save_dir)

        img = load_image(sample_data_dir, int(h.strip()), int(w.strip()))
        img = IMAGE_TRANSFORMS(img).to(dtype=vae_dtype).unsqueeze(0)

        with torch.no_grad():
            img = img.to(accelerator.device)
            # (1) encoder make latent
            latent = DiagonalGaussianDistribution(student(img)).sample()
            # (2) decoder
            recon_img = vae.decode(latent)['sample']
            recon_img = (recon_img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (recon_img * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(save_dir)

    print(f'\n step 3. inference')
    print(' (3.1) anormal test')
    if args.training_data_check :

        save_dir = os.path.join(args.output_dir, 'inference/vae_result_check/training_dataset')
        os.makedirs(save_dir, exist_ok=True)
        save_base_dir = os.path.join(save_dir, f'student_epoch_{student_epoch}')
        os.makedirs(save_base_dir, exist_ok=True)


        anormal_folder = os.path.join(args.anormal_folder, 'test/bad')
        classes = os.listdir(anormal_folder)
        for class_ in classes:

            class_dir = os.path.join(anormal_folder, class_)
            class_mask_dir = None

            class_save_dir = os.path.join(save_base_dir, class_)
            os.makedirs(class_save_dir, exist_ok=True)

            images = os.listdir(class_dir)

            for image in images:
                image_dir = os.path.join(class_dir, image)
                mask_dir = None
                name, ext = os.path.splitext(image)
                img_save_dir = os.path.join(class_save_dir, f'{name}_recon.png')
                compare_save_dir = os.path.join(class_save_dir, image)
                mask_save_dir = None
                recon(mask_dir, image_dir, mask_save_dir, img_save_dir, compare_save_dir)

    else :

        save_dir = os.path.join(args.output_dir, 'inference/vae_result_check/test_dataset')
        os.makedirs(save_dir, exist_ok=True)
        save_base_dir = os.path.join(save_dir, f'student_epoch_{student_epoch}')
        os.makedirs(save_base_dir, exist_ok=True)

        anormal_folder = os.path.join(args.anormal_folder, 'train')
        anormal_mask_folder = os.path.join(args.anormal_folder, 'train_mask')
        classes = os.listdir(anormal_folder)
        for class_ in classes:

            class_dir = os.path.join(anormal_folder, class_)
            class_mask_dir = os.path.join(anormal_mask_folder, class_)

            if class_ == 'good' :
                class_dir = os.path.join(anormal_folder, f'{class_}/rgb')
                class_mask_dir = os.path.join(anormal_folder, f'{class_}/gt')

            class_save_dir = os.path.join(save_base_dir, class_)
            os.makedirs(class_save_dir, exist_ok=True)

            images = os.listdir(class_dir)
            mask_images = os.listdir(class_mask_dir)

            for image in images :

                image_dir = os.path.join(class_dir, image)
                mask_dir = os.path.join(class_mask_dir, image)
                name, ext = os.path.splitext(image)

                img_save_dir = os.path.join(class_save_dir, f'{name}_recon.png')
                compare_save_dir = os.path.join(class_save_dir, image)
                mask_save_dir = os.path.join(class_save_dir, f'{name}_mask.png')
                recon(mask_dir, image_dir, mask_save_dir, img_save_dir, compare_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    train_util.add_sd_models_arguments(parser)
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    config_util.add_config_arguments(parser)
    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）", )
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--resolution", type=str, default=None, help="resolution in training ('size' or 'width,height') ", )
    parser.add_argument("--logging_dir",type=str,default=None,)
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"],)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int,default=1,)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--full_fp16", action="store_true",
                        help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients / 勾配も含めてbf16で学習する")
    parser.add_argument("--save_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--lowram",action="store_true",
                        help="enable low RAM optimization. e.g. load models to VRAM instead of RAM ",)
    parser.add_argument("--vae", type=str, default=None,
                        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")
    parser.add_argument("--anormal_sample_data_dir", type=str,
                        default=r'../../../MyData/anomaly_detection/VisA/MVTecAD/bagel/test/crack/rgb/000.png')
    parser.add_argument("--anormal_folder", type=str,)
    parser.add_argument("--normal_sample_data_dir", type=str,
                        default=r'../../../MyData/anomaly_detection/VisA/MVTecAD/bagel/test/good/rgb/000.png')
    parser.add_argument("--student_pretrained_dir", type=str,
                        default='../result/MVTec_experiment/bagel/vae_training/vae_model/vae_epoch_000005/pytorch_model.bin')
    parser.add_argument("--training_data_check", action="store_true",)
    args = parser.parse_args()
    main(args)

