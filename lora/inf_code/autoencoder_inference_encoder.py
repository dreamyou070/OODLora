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
from STTraining import Encoder_Teacher, Encoder_Student, Decoder_Student
from library.model_util import create_vae_diffusers_config, convert_ldm_vae_checkpoint, load_checkpoint_with_text_encoder_conversion
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None
from accelerate import Accelerator
from diffusers import AutoencoderKL


def main(args):

    print(f'\n step 1. setting')
    print(f' (1) session')
    if args.process_title:
        setproctitle(args.process_title)
    else:
        setproctitle('parksooyeon')

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
    print(f' (2) vae pretrained')
    name_or_path = args.pretrained_model_name_or_path
    vae_config = create_vae_diffusers_config()
    _, state_dict = load_checkpoint_with_text_encoder_conversion(name_or_path,
                                                                 device='cpu')
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)
    vae = AutoencoderKL(**vae_config)
    info = vae.load_state_dict(converted_vae_checkpoint)
    vae.to(dtype=weight_dtype, device=accelerator.device)

    print(f' (3) vae student encoder')
    student_vae = AutoencoderKL.from_config(vae.config)
    student_vae_encoder = student_vae.encoder
    student_vae_encoder_quantize = student_vae.quant_conv
    student_encoder = Encoder_Student(student_vae_encoder, student_vae_encoder_quantize)

    print(f' (4) vae student decoder')
    student_vae_decoder = student_vae.decoder
    student_vae_decoder_quantize = student_vae.post_quant_conv
    student_decoder = Decoder_Student(student_vae_decoder, student_vae_decoder_quantize)

    print(f' (5) student model loading')
    def get_state_dict(dir) :
        model_state_dict = torch.load(dir, map_location="cpu")
        state_dict = {}
        for k, v in model_state_dict.items():
            k_ = '.'.join(k.split('.')[1:])
            state_dict[k_] = v
        return state_dict

    if args.student_encoder_pretrained_dir is not None:
        student_models = os.listdir(args.student_encoder_pretrained_dir)
        encoder_state_dict = get_state_dict(args.student_encoder_pretrained_dir)
        student_encoder.load_state_dict(encoder_state_dict, strict=True)
    if args.student_decoder_pretrained_dir is not None:
        decoder_state_dict = get_state_dict(args.student_decoder_pretrained_dir)
        student_decoder.load_state_dict(decoder_state_dict, strict=True)

    student_encoder.requires_grad_(False)
    student_encoder.eval()
    student_encoder.to(accelerator.device, dtype=vae_dtype)
    student_decoder.requires_grad_(False)
    student_decoder.eval()
    student_decoder.to(accelerator.device, dtype=vae_dtype)

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
            if args.student_encoder_pretrained_dir is not None:
                latent = DiagonalGaussianDistribution(student_encoder(img)).sample()
            else :
                latent = DiagonalGaussianDistribution(vae.encode(img)).sample()

            # (2) decoder
            if args.student_decoder_pretrained_dir is not None:
                recon_img = student_decoder(latent)
            else :
                recon_img = vae.decode(latent)['sample']
            recon_img = (recon_img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (recon_img * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(save_dir)

    print(f'\n step 3. inference')
    print(' (3.1) anormal test')
    if args.training_data_check :
        save_base_dir = os.path.join(args.output_dir, 'inference/vae_result_check/training_dataset')
        os.makedirs(save_base_dir, exist_ok=True)
        anormal_folder = os.path.join(args.anormal_folder, 'train/bad')
        mask_folder = os.path.join(args.anormal_folder, 'train/gt')
        classes = os.listdir(anormal_folder)
        for class_ in classes:

            class_dir = os.path.join(anormal_folder, class_)
            if '_' in class_:
                class__ = class_.split('_')[1:]
                class__ = '_'.join(class__)
            class_mask_dir = os.path.join(mask_folder, class__)

            class_save_dir = os.path.join(save_base_dir, class__)
            os.makedirs(class_save_dir, exist_ok=True)

            images = os.listdir(class_dir)

            for i, image in enumerate(images) :
                if i < 5 :
                    image_dir = os.path.join(class_dir, image)
                    if 'good' not in class_ :
                        mask_dir = os.path.join(class_mask_dir, image)
                    else :
                        mask_dir = None
                    name, ext = os.path.splitext(image)
                    img_save_dir = os.path.join(class_save_dir, f'{name}_recon.png')
                    compare_save_dir = os.path.join(class_save_dir, image)
                    if 'good' not in class_ :
                        mask_save_dir = os.path.join(class_save_dir, f'{name}_gt.png')
                    else :
                        mask_save_dir = None
                    recon(mask_dir, image_dir, mask_save_dir, img_save_dir, compare_save_dir)

    else :

        save_base_dir = os.path.join(args.output_dir, 'inference/vae_result_check/test_dataset')
        os.makedirs(save_base_dir, exist_ok=True)

        anormal_folder = os.path.join(args.anormal_folder, 'test/rgb')
        anormal_mask_folder = os.path.join(args.anormal_folder, 'test/gt')
        classes = os.listdir(anormal_folder)
        for class_ in classes:
            class_dir = os.path.join(anormal_folder, class_)
            class_mask_dir = os.path.join(anormal_mask_folder, class_)
            if 'good' in class_ :
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
    parser.add_argument("--student_encoder_pretrained_dir", type=str,)
    parser.add_argument("--student_decoder_pretrained_dir", type=str, )
    parser.add_argument("--training_data_check", action="store_true",)
    args = parser.parse_args()
    main(args)

