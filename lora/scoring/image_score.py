import argparse
import os, sys
import random
from accelerate.utils import set_seed
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
import torch
import cv2
from attention_store import AttentionStore
import sys, importlib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score,auc,average_precision_score
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
from diffusers import (DDPMScheduler,EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler,DPMSolverSinglestepScheduler,
                       LMSDiscreteScheduler,PNDMScheduler,EulerDiscreteScheduler,HeunDiscreteScheduler,
                       KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler)
from diffusers import DDIMScheduler

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def load_64(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:,:]
    else:
        image = image_path
    h, w = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w= image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    new_pil = Image.fromarray(image).resize((64,64))
    image = np.array(new_pil)
    return image
def image2latent(image, vae, device, weight_dtype):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device, weight_dtype)
            latents = vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def main(args) :

    print(f' \n step 1. setting')
    print(f' (1) process title')
    if args.process_title:
        setproctitle(args.process_title)
    else:
        setproctitle('parksooyeon')

    print(f' (2) seed')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f' (3) weight dtype')
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    print(f" (4) save dir")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f' (5) device')
    device = args.device

    print(f' \n step 2. make stable diffusion model')
    print(f' (2.1) tokenizer')
    tokenizer = train_util.load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
    print(f' (2.2) SD')
    text_encoder, vae, unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype, device,
                                                                                          unet_use_linear_projection_in_v2=False, )
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    print(f' (2.3) scheduler')
    sched_init_args = {}
    if args.sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif args.sample_sampler == "ddpm":
        scheduler_cls = DDPMScheduler
    elif args.sample_sampler == "pndm" :
        scheduler_cls = PNDMScheduler
    elif args.sample_sampler == "lms" or args.sample_sampler == "k_lms" :
        scheduler_cls = LMSDiscreteScheduler
    elif args.sample_sampler == "euler" or args.sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_sampler == "euler_a" or args.sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_sampler == "dpmsolver" or args.sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sample_sampler
    elif args.sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_sampler == "dpm_2" or args.sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif args.sample_sampler == "dpm_2_a" or args.sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else :
        scheduler_cls = DDIMScheduler
    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction "
    SCHEDULER_LINEAR_START = 0.00085
    SCHEDULER_LINEAR_END = 0.0120
    SCHEDULER_TIMESTEPS = 1000
    SCHEDLER_SCHEDULE = "scaled_linear"
    scheduler = scheduler_cls(num_train_timesteps=SCHEDULER_TIMESTEPS, beta_start=SCHEDULER_LINEAR_START,
                              beta_end=SCHEDULER_LINEAR_END, beta_schedule=SCHEDLER_SCHEDULE,)
    scheduler.set_timesteps(args.num_ddim_steps)
    inference_times = scheduler.timesteps
    print(f' (2.4) model to accelerator device')
    if len(text_encoders) > 1:
        unet, t_enc1, t_enc2, vae = unet.to(device), text_encoders[0].to(device), text_encoders[1].to(device), vae.to(device)
        text_encoder = [t_enc1, t_enc2]
        del t_enc1, t_enc2
    else:
        unet, text_encoder, vae = unet.to(device), text_encoder.to(device), vae.to(device)
        text_encoders = [text_encoder]
    print(f' (2.5) network')
    sys.path.append(os.path.dirname(__file__))
    network_module = importlib.import_module(args.network_module)
    print(f' (1.3.1) merging weights')
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value
    print(f' (1.3.3) make network')
    if args.dim_from_weights:
        network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet,
                                                                **net_kwargs)
    else:
        network = network_module.create_network(1.0,
                                                args.network_dim,
                                                args.network_alpha,
                                                vae, text_encoder, unet, neuron_dropout=args.network_dropout,
                                                **net_kwargs, )
    print(f' (1.3.4) apply trained state dict')
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
    network.to(device)

    print(f' \n step 3. image scoring')
    orgin_img_dir = '../examples/013_origin.png'
    orgin_np = load_512(orgin_img_dir)
    orgin_latent = image2latent(orgin_np, vae, device, weight_dtype)  # 1,4,64,64
    #orgin_latent = torch.flatten(orgin_latent, start_dim=1)
    #orgin_latent_np = orgin_latent.detach().cpu().numpy()

    recon_img_dir = '../examples/013_recon.png'
    recon_np = load_512(recon_img_dir)
    recon_latent = image2latent(recon_np, vae, device, weight_dtype)
    #recon_latent = torch.flatten(recon_latent, start_dim=1)
    #recon_latent_np = recon_latent.detach().cpu().numpy()

    diff_latent = torch.nn.functional.mse_loss(orgin_latent, recon_latent, reduction='none')
    batch_size, h, w = diff_latent.shape[0],diff_latent.shape[-2],diff_latent.shape[-1]
    anomal_map = torch.sum(diff_latent, dim=1)
    anomal_vector = torch.flatten(anomal_map, start_dim=1)
    max_value = torch.max(anomal_vector, dim=1)[0].unsqueeze(1)
    normalized_anomal_vector = anomal_vector / max_value
    normalized_anomal_map = normalized_anomal_vector.view(batch_size, 1, h, w)
    for index in range(batch_size) :
        anomal_map = normalized_anomal_map[index]
        ano_map = cvt2heatmap(anomal_map * 255.0)
        print(f'ano_map : {ano_map}')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # step 1. setting
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="use mixed precision / 混合精度を使う場合、その精度")
    parser.add_argument("--save_precision", type=str, choices=["fp16", "bf16", 'float'], default = 'fp16')
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う", )
    parser.add_argument("--output_dir", type=str, )
    parser.add_argument("--device", type=str, default="cuda")

    # step 2. model
    train_util.add_sd_models_arguments(parser)
    parser.add_argument("--vae", type=str, default=None,
                        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")
    parser.add_argument("--sample_sampler",type=str,default="ddim",
                        choices=["ddim","pndm","lms","euler","euler_a","heun","dpm_2","dpm_2_a","dpmsolver",
                                 "dpmsolver++","dpmsingle","k_lms","k_euler","k_euler_a","k_dpm_2","k_dpm_2_a",],
                        help=f"sampler (scheduler) type for sample images / サンプル出力時のサンプラー（スケジューラ）の種類",)
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--base_weights", type=str, default=None, nargs="*",
                        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル", )
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*",
                        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率", )
    parser.add_argument("--network_dim", type=int, default=None,
                        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）")
    parser.add_argument("--network_alpha", type=float, default=1,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)", )
    parser.add_argument("--network_dropout", type=float, default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)", )
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value) / ネットワークへの追加の引数")
    parser.add_argument("--dim_from_weights", action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する", )
    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    args = parser.parse_args()
    main(args)