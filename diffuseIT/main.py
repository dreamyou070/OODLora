from optimization.image_editor import ImageEditor
import argparse
# python main.py --use_range_restart --use_noise_aug_all --regularize_content --init_image input_example/lion1.jpg
import os
from utils_visualize.metrics_accumulator import MetricsAccumulator
from numpy import random
from optimization.augmentations import ImageAugmentations
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss
import numpy as np
from src.vqc_core import *
from model_vit.loss_vit import Loss_vit
from guided_diffusion.guided_diffusion.script_util import (create_model_and_diffusion, model_and_diffusion_defaults,)
from pathlib import Path
from id_loss import IDLoss
from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer
from utils_flexit.torch_utils import TensorDict

mean_sig = lambda x: sum(x) / len(x)

class CLIPS:
    def __init__(self, names=['RN50x4', 'ViT-B/32'], device='cpu', **kwargs):
        self.networks = {n: CLIP(n, device=device, **kwargs) for n in names}

    def encode_image(self, x, ncuts=0):
        return TensorDict({name: model.encode_image(x, ncuts=ncuts) for name, model in self.networks.items()})

    def encode_text(self, x):
        return TensorDict({name: model.encode_text(x) for name, model in self.networks.items()})


class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(self.args.output_path)
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()

        if self.args.use_ffhq:
            self.model_config.update({"attention_resolutions": "16",
                                      "class_cond": self.args.model_output_size == 512,
                                      "diffusion_steps": 1000,
                                      "rescale_timesteps": True,
                                      "timestep_respacing": self.args.timestep_respacing,
                                      "image_size": self.args.model_output_size,
                                      "learn_sigma": True,
                                      "noise_schedule": "linear",
                                      "num_channels": 128,
                                      "num_head_channels": 64,
                                      "num_res_blocks": 1,
                                      "resblock_updown": True,"use_fp16": False,
                                      "use_scale_shift_norm": True,})
        else:
            print(f' get model config ...')
            self.model_config.update({"attention_resolutions": "32, 16, 8",
                                      "class_cond": self.args.model_output_size == 512,
                                      "diffusion_steps": 1000,
                                      "rescale_timesteps": True,
                                      "timestep_respacing": self.args.timestep_respacing,
                                      "image_size": self.args.model_output_size,
                                      "learn_sigma": True,
                                      "noise_schedule": "linear",
                                      "num_channels": 256,
                                      "num_head_channels": 64,
                                      "num_res_blocks": 2,
                                      "resblock_updown": True,
                                      "use_fp16": True,
                                      "use_scale_shift_norm": True,})
        print(f' loading model (model and ddpm diffusino model)')
        self.device = torch.device( f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f' (1) ddpm model')
        print(f' (2) model')
        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        if self.args.use_ffhq:
            self.model.load_state_dict(torch.load("../../checkpoints/ffhq_10m.pt", map_location="cpu", ))
            self.idloss = IDLoss().to(self.device)
        else:
            self.model.load_state_dict(torch.load("../../checkpoints/256x256_diffusion_uncond.pt" if self.args.model_output_size == 256
                                                  else "checkpoints/512x512_diffusion.pt",map_location="cpu",))
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        print(f' (3) vit model')
        with open("model_vit/config.yaml", "r") as ff:
            config = yaml.safe_load(ff)
        cfg = config
        # -------------------------------------------------------- VIT -------------------------------------------------------- #
        self.VIT_LOSS = Loss_vit(cfg,
                                 lambda_ssim=self.args.lambda_ssim,               # (2)
                                 lambda_dir_cls=self.args.lambda_dir_cls,
                                 lambda_contra_ssim=self.args.lambda_contra_ssim, # (1)
                                 lambda_trg=args.lambda_trg).eval()  # .requires_grad_(False)

        # ---------------------------------------------------------------------------------------------------------------- #
        print(f' (4) clip model')
        # init networks
        if self.args.target_image is None:
            self.clip_net = CLIPS(names=self.args.clip_models,
                                  device=self.device,
                                  erasing=False)  # .requires_grad_(False)
        self.cm = ColorMatcher()
        self.clip_size = 224
        self.clip_normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        #         self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)
        print(f' (5) image preprocessing')
        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()

    def noisy_aug(self, t, x, x_hat):
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        x_mix = x_hat * fac + x * (1 - fac)
        return x_mix

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def edit_image_by_prompt(self):

        print(f' (1) init src image ...')
        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1))

        print(f' (2) check trg image')
        self.target_image = None
        if self.args.target_image is not None:
            print(f'There is Trg Image')
            self.target_image_pil = Image.open(self.args.target_image).convert("RGB")
            self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.target_image = (TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1))
        else :
            print('No Trg Image')
        self.prev = self.init_image.detach()
        if self.target_image is None:
            txt2 = self.args.prompt
            txt1 = self.args.source
            with torch.no_grad():
                self.E_I0 = E_I0 = self.clip_net.encode_image(0.5 * self.init_image + 0.5, ncuts=0)
                self.E_S, self.E_T = E_S, E_T = self.clip_net.encode_text([txt1, txt2])
                self.tgt = (1 * E_T - 0.4 * E_S + 0.2 * E_I0).normalize()
            pred = self.clip_net.encode_image(0.5 * self.prev + 0.5, ncuts=0)
            clip_loss = - (pred @ self.tgt.T).flatten().reduce(mean_sig)
            self.loss_prev = clip_loss.detach().clone()
        self.flag_resample = False
        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

        def cond_fn(x, t, y=None):
            if self.args.prompt == "":
                return torch.zeros_like(x)
            self.flag_resample = False
            with torch.enable_grad():
                frac_cont = 1.0
                if self.target_image is None:
                    if self.args.use_prog_contrast:
                        if self.loss_prev > -0.5:
                            frac_cont = 0.5
                        elif self.loss_prev > -0.4:
                            frac_cont = 0.25
                    if self.args.regularize_content:
                        if self.loss_prev < -0.5:
                            frac_cont = 2
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance( self.model, x, t, clip_denoised=False, model_kwargs={"y": y})

                loss = torch.tensor(0)
                if self.target_image is None:
                    if self.args.clip_guidance_lambda != 0:
                        x_clip = self.noisy_aug(t[0].item(), x, out["pred_xstart"])
                        pred = self.clip_net.encode_image(0.5 * x_clip + 0.5, ncuts=self.args.aug_num)
                        clip_loss = - (pred @ self.tgt.T).flatten().reduce(mean_sig)
                        loss = loss + clip_loss * self.args.clip_guidance_lambda
                        self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
                        self.loss_prev = clip_loss.detach().clone()
                if self.args.use_noise_aug_all:
                    x_in = self.noisy_aug(t[0].item(), x, out["pred_xstart"])
                else:
                    x_in = out["pred_xstart"]

                if self.args.vit_lambda != 0:

                    if t[0] > self.args.diff_iter:
                        vit_loss, vit_loss_val = self.VIT_LOSS(x_in, self.init_image, self.prev, use_dir=True,
                                                               frac_cont=frac_cont, target=self.target_image)
                    else:
                        vit_loss, vit_loss_val = self.VIT_LOSS(x_in, self.init_image, self.prev, use_dir=False,
                                                               frac_cont=frac_cont, target=self.target_image)
                    loss = loss + vit_loss

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())
                if self.target_image is not None:
                    loss = loss + mse_loss(x_in, self.target_image) * self.args.l2_trg_lambda

                if self.args.use_ffhq:
                    loss = loss + self.idloss(x_in, self.init_image) * self.args.id_lambda
                self.prev = x_in.detach().clone()

                if self.args.use_range_restart:
                    if t[0].item() < total_steps:
                        if self.args.use_ffhq:
                            if r_loss > 0.1:
                                self.flag_resample = True
                        else:
                            if r_loss > 0.01:
                                self.flag_resample = True

            return -torch.autograd.grad(loss, x)[0], self.flag_resample

        print(f' (3) Iteratively denoising (image translatino through text)')
        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number} with p_sample_loop_progressive")
            print(f'args.ddim : {self.args.ddim}')
            sample_func = (self.diffusion.ddim_sample_loop_progressive  if self.args.ddim else self.diffusion.p_sample_loop_progressive)
            
            samples = sample_func(self.model,
                                  (self.args.batch_size,3,self.model_config["image_size"],self.model_config["image_size"],),
                                  clip_denoised=False,
                                  model_kwargs={} if self.args.model_output_size == 256 else {"y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)},
                                  cond_fn=cond_fn,
                                  progress=True,
                                  skip_timesteps=self.args.skip_timesteps,
                                  init_image=self.init_image,
                                  postprocess_fn=None,
                                  randomize_class=True,)
            """
            if self.flag_resample:
                continue
            """
            """
            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
            total_steps_with_resample = self.diffusion.num_timesteps - self.args.skip_timesteps - 1 + (self.args.resample_num - 1)
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps_with_resample
                for b in range(self.args.batch_size):
                    pred_image = sample["pred_xstart"][b]
                    visualization_path = Path(os.path.join(self.args.output_path, self.args.output_file) )
                    visualization_path = visualization_path.with_name(
                        f"{visualization_path.stem}_i_{iteration_number}_b_{b}{visualization_path.suffix}")
                    pred_image = pred_image.add(1).div(2).clamp(0, 1)
                    pred_image_pil = TF.to_pil_image(pred_image)
            ranked_pred_path = self.ranked_results_path / (visualization_path.name)
            if self.args.target_image is not None:
                if self.args.use_colormatch:
                    src_image = Normalizer(np.asarray(pred_image_pil)).type_norm()
                    trg_image = Normalizer(np.asarray(self.target_image_pil)).type_norm()
                    img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                    img_res = Normalizer(img_res).uint8_norm()
                    save_img_file(img_res, str(ranked_pred_path))
            else:
                pred_image_pil.save(ranked_pred_path)
            """


def main(args) :

    print(f'step 1. get image editor')
    image_editor = ImageEditor(args)

    print(f'step 2. image editing')
    image_editor.edit_image_by_prompt()
    # image_editor.reconstruct_image()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Inputs
    parser.add_argument("-p", "--prompt", default ='Black Leopard',
                        type=str, help="The prompt for the desired editing", required=False)
    parser.add_argument("-s", "--source", type=str, default='Lion',
                        help="The prompt for the source image", required=False)
    parser.add_argument("-i", "--init_image", type=str, default="input_example/lion1.jpg",
                        help="The path to the source image input", required=True)
    parser.add_argument("-tg", "--target_image", type=str, help="The path to the target style image", required=False)
    parser.add_argument("--skip_timesteps",type=int,help="How many steps to skip during the diffusion.",default=40,)
    parser.add_argument("--ddim",help="Indicator for using DDIM instead of DDPM",action="store_true",)
    parser.add_argument("--timestep_respacing",type=str,help="How to respace the intervals of the diffusion process (number between 1 and 1000).",default="100",)
    parser.add_argument( "--model_output_size",        type=int,        help="The resolution of the outputs of the diffusion model",
        default=256,        choices=[256, 512],    )
    parser.add_argument("--clip_models",help="List for CLIP models",                        nargs="+",default=['RN50', 'RN50x4', 'ViT-B/32', 'RN50x16', 'ViT-B/16'],)
    # Augmentations
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=8)
    parser.add_argument("--diff_iter", type=int, help="The number of augmentation", default=50)
    parser.add_argument( "--clip_guidance_lambda",        type=float,        help="Controls how much the image should look like the prompt",
        default=2000,)
    parser.add_argument("--lambda_trg",        type=float,        help="style loss for target style image",        default=2000,    )
    parser.add_argument("--l2_trg_lambda",        type=float,        help="l2 loss for target style image",        default=3000,    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=200,
    )
    parser.add_argument("--vit_lambda", type=float, help="total vit loss", default=1,)
    parser.add_argument("--lambda_ssim", type=float, help="key self similarity loss", default=1000,)
    parser.add_argument("--lambda_dir_cls", type=float, help="semantic divergence loss", default=100,)
    parser.add_argument("--lambda_contra_ssim", type=float, help="contrastive loss for keys", default=200,)
    parser.add_argument(
        "--id_lambda", type=float, help="identity loss", default=100,)
    parser.add_argument("--resample_num", type=float, help="resampling number", default=10,)
    parser.add_argument("--seed", type=int, help="The random seed", default=None)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("-o","--output_file",type=str,help="The filename to save, must be png",default="./outputs/output_leopard",)
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=10)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=1,)

    parser.add_argument("--use_ffhq",action="store_true",)
    parser.add_argument(
        "--use_prog_contrast",
        action="store_true",
    )
    parser.add_argument(
        "--use_range_restart",
        action="store_true",
    )
    parser.add_argument(
        "--use_colormatch",
        action="store_true",
    )
    parser.add_argument(
        "--use_noise_aug_all",
        action="store_true",
    )
    parser.add_argument(
        "--regularize_content",
        action="store_true",    )
    args = parser.parse_args()
    main(args)