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
    accelerator = train_util.prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f'\n step 2. model')
    print(f' (1) mixed precision and model')
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
    print(f' (2) model')
    text_encoder, vae, unet, _ = train_util.load_target_model(weight_dtype, accelerator)
    # lr schedulerを用意する
    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=3,
                                       out_channels=3,kernel_size=4, activation=(Act.LEAKYRELU, {"negative_slope": 0.2, }),
                                       norm="BATCH", bias=False, padding=1, )

    print(f' (3) loading state')
    vae_pretrained_dir = r'/data7/sooyeon/result/MVTec_experiment/bagel/vae_training/'
    discriminator_pretrained_dir = r'/data7/sooyeon/result/MVTec_experiment/bagel/vae_training'





    vae.requires_grad_(True)
    vae.train()
    vae.to(dtype=vae_dtype)
    #vae.to(dtype=weight_dtype)
    discriminator.requires_grad_(True)
    discriminator.train()
    discriminator.to(dtype=weight_dtype)

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    perceptual_weight = 0.001


    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1


    if args.sample_every_n_epochs is not None and epoch+1 % args.sample_every_n_epochs == 0 :
        print('sampling')
        sample_data_dir = r'../../../MyData/anomaly_detection/VisA/MVTecAD/bagel/test/crack/rgb/000.png'
        h,w = args.resolution
        img = load_image(sample_data_dir, int(h), int(w))
        img = IMAGE_TRANSFORMS(img).to(dtype=vae_dtype).unsqueeze(0)
        with torch.no_grad():
            if accelerator.is_main_process:
                inf_vae = accelerator.unwrap_model(vae).to(dtype=vae_dtype)
                img = img.to(inf_vae.device)
                recon_img = inf_vae(img).sample
                recon_img = (recon_img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (recon_img * 255).astype(np.uint8)
                image = Image.fromarray(image)
                save_dir = os.path.join(args.output_dir, 'sample')
                os.makedirs(save_dir, exist_ok=True)
                image.save(os.path.join(save_dir, f'anormal_recon_epoch_{epoch}.png'))
        sample_data_dir = r'../../../MyData/anomaly_detection/VisA/MVTecAD/bagel/test/good/rgb/000.png'
        img = load_image(sample_data_dir, int(h), int(w))
        img = IMAGE_TRANSFORMS(img).to(dtype=vae_dtype).unsqueeze(0)
        with torch.no_grad():
            if accelerator.is_main_process:
                inf_vae = accelerator.unwrap_model(vae).to(dtype=vae_dtype)
                img = img.to(inf_vae.device)
                recon_img = inf_vae(img).sample
                recon_img = (recon_img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (recon_img * 255).astype(np.uint8)
                image = Image.fromarray(image)
                save_dir = os.path.join(args.output_dir, 'sample')
                os.makedirs(save_dir, exist_ok=True)
                image.save(os.path.join(save_dir, f'normal_recon_epoch_{epoch}.png'))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    # parser.add_argument("--wandb_init_name", type=str)
    parser.add_argument("--wandb_log_template_path", type=str)
    parser.add_argument("--wandb_key", type=str)

    # step 2. dataset
    train_util.add_dataset_arguments(parser, True, True, True)
    parser.add_argument("--mask_dir", type=str, default='')
    parser.add_argument("--class_caption", type=str, default='')
    parser.add_argument("--no_metadata", action="store_true",
                        help="do not save metadata in output model / メタデータを出力先モデルに保存しない")

    # step 3. model
    train_util.add_sd_models_arguments(parser)
    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--network_dim", type=int, default=None,
                        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）")
    parser.add_argument("--network_alpha", type=float, default=1,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)", )
    parser.add_argument("--network_dropout", type=float, default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)", )
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value) / ネットワークへの追加の引数")

    # step 4. training
    train_util.add_training_arguments(parser, True)
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")

    # step 5. optimizer
    train_util.add_optimizer_arguments(parser)

    config_util.add_config_arguments(parser)

    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）", )
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
    parser.add_argument("--dim_from_weights", action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する", )
    parser.add_argument("--scale_weight_norms", type=float, default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. ", )
    parser.add_argument("--base_weights", type=str, default=None, nargs="*",
                        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル", )
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*",
                        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う", )
    parser.add_argument("--net_key_names", type=str, default='text')
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--contrastive_eps", type=float, default=0.00005)
    # class_caption
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    main(args)