import argparse
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.networks.layers import Act
from torch.nn import L1Loss
import math
import os
from torch.cuda.amp import GradScaler, autocast
import random
import time
import json
from multiprocessing import Value
from tqdm import tqdm
import toml
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import model_util
import library.train_util as train_util
from library.train_util import (DreamBoothDataset, )
import library.config_util as config_util
from library.config_util import (ConfigSanitizer, BlueprintGenerator, )
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import prepare_scheduler_for_custom_training
import torch
from utils.image_utils import load_image, latent2image, IMAGE_TRANSFORMS
import numpy as np
from PIL import Image
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from library.ipex import ipex_init
        ipex_init()
except Exception:
    pass
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
    is_main_process = accelerator.is_main_process

    print(f'\n step 2. model')
    print(f' (1) mixed precision and model')
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32
    print(f' (2) model')
    text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
    # lr schedulerを用意する
    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=3,
                                       out_channels=3,kernel_size=4, activation=(Act.LEAKYRELU, {"negative_slope": 0.2, }),
                                       norm="BATCH", bias=False, padding=1, )

    print(f' (3) loading state')
    vae_pretrained_dir = '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/vae_training/vae_model/vae_epoch_000001/pytorch_model.bin'
    discriminator_pretrained_dir = '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/vae_training/discriminator_model/discriminator_epoch_000001/pytorch_model.bin'
    vae.load_state_dict(torch.load(vae_pretrained_dir))
    discriminator.load_state_dict(torch.load(discriminator_pretrained_dir))

    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device, dtype=vae_dtype)

    discriminator.requires_grad_(False)
    discriminator.eval()
    discriminator.to(accelerator.device, dtype=vae_dtype)
    #discriminator.to(args.device)

    print(f'\n step 3. inference')
    print('sampling')
    sample_data_dir = r'../../../MyData/anomaly_detection/VisA/MVTecAD/bagel/test/crack/rgb/000.png'
    print(f'args.resolution: {args.resolution}')
    h,w = args.resolution.split(',')[0], args.resolution.split(',')[1]
    img = load_image(sample_data_dir, int(h.strip()), int(w.strip()))
    img = IMAGE_TRANSFORMS(img).to(dtype=vae_dtype).unsqueeze(0)
    with torch.no_grad():
        img = img.to(vae.device)
        recon_img = vae(img).sample
        recon_img = (recon_img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (recon_img * 255).astype(np.uint8)
        image = Image.fromarray(image)
        save_dir = os.path.join(args.output_dir, 'test')
        os.makedirs(save_dir, exist_ok=True)
        image.save(os.path.join(save_dir, f'anormal_recon_test.png'))


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
    args = parser.parse_args()
    main(args)