import torch
import numpy as np
from typing import Union
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_utils import call_unet
from image_utils import latent2image
from PIL import Image
from image_utils import image2latent
from diffusers import (DDPMScheduler,EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler,DPMSolverSinglestepScheduler,
                       LMSDiscreteScheduler,PNDMScheduler,EulerDiscreteScheduler,HeunDiscreteScheduler,
                       KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler, DDIMScheduler)

def get_scheduler(sampler, v_parameterization):
    sched_init_args = {}
    if sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif sampler == "ddpm":
        scheduler_cls = DDPMScheduler
    elif sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif sampler == "lms" or sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif sampler == "euler" or sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif sampler == "euler_a" or sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sampler == "dpmsolver" or sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sampler
    elif sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif sampler == "dpm_2" or sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif sampler == "dpm_2_a" or sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler
    if v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction "
    return scheduler_cls, sched_init_args

def next_step(model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray],
              scheduler):
    timestep, next_timestep = timestep, min( timestep + scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999)
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_matrix = torch.ones_like(model_output) * alpha_prod_t
    alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
    alpha_prod_t_next_matrix = torch.ones_like(model_output) * alpha_prod_t_next
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_matrix = torch.ones_like(model_output) * beta_prod_t
    next_original_sample = (sample - beta_prod_t_matrix ** 0.5 * model_output) / alpha_prod_t_matrix ** 0.5
    next_sample_direction = (torch.ones_like(model_output) - alpha_prod_t_next_matrix) ** 0.5 * model_output
    next_sample = alpha_prod_t_next_matrix ** 0.5 * next_original_sample + next_sample_direction
    return next_sample

def prev_step(model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray],
              scheduler):
    timestep, prev_timestep = timestep, max( timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 0)
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_matrix = torch.ones_like(model_output) * alpha_prod_t

    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]

    alpha_prod_t_prev_matrix = torch.ones_like(model_output) * alpha_prod_t_prev
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_matrix = torch.ones_like(model_output) * beta_prod_t
    prev_original_sample = (sample - beta_prod_t_matrix ** 0.5 * model_output) / alpha_prod_t_matrix ** 0.5
    prev_sample_direction = (torch.ones_like(model_output) - alpha_prod_t_prev_matrix) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev_matrix ** 0.5 * prev_original_sample + prev_sample_direction
    return prev_sample

def pred_x0 (model_output, timestep, sample, scheduler) :
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    a = sample / ((alpha_prod_t)**0.5)
    b = (((1-alpha_prod_t)/alpha_prod_t)**0.5) * model_output
    return a - b

@torch.no_grad()
def ddim_loop(args, latent, context, inference_times, scheduler, unet, vae, final_time, base_folder_dir, name,):
    if context.shape[0] == 1:
        cond_embeddings = context
    else :
        uncond_embeddings, cond_embeddings = context.chunk(2)
    time_steps = []
    latent = latent.clone().detach()
    latent_dict = {}
    latent_dict[0] = latent
    pil_images = []
    flip_times = inference_times
    for i, t in enumerate(flip_times[:-1]):
        next_time = flip_times[i + 1]
        if next_time <= final_time :
            latent_dict[int(t)] = latent
            time_steps.append(t)
            noise_pred = call_unet(unet, latent, t, cond_embeddings, None, None)
            latent = next_step(noise_pred, int(t), latent, scheduler)
            #np_img = latent2image(latent, vae, return_type='np')
            #pil_img = Image.fromarray(np_img)
            #pil_images.append(pil_img)
            #pil_img.save(os.path.join(base_folder_dir, f'{name}_noising_{next_time}.png'))
        else :
            time_steps.append(final_time)
            latent_dict[int(final_time)] = latent
            break
    return latent_dict, time_steps, pil_images



@torch.no_grad()
def recon_loop(args, z_latent_dict, start_latent, gt_pil, context, inference_times, scheduler, unet, vae, base_folder_dir, controller, name,weight_dtype):
    if context.shape[0] == 2:
        uncon, con = context.chunk(2)
    else:
        con = context
    # inference_times = [100,80, ... 0]
    x_latent = start_latent
    time_steps = []
    pil_images = []
    x_latent_dict = {}
    x_latent_dict[inference_times[0]] = x_latent

    for i, t in enumerate(inference_times[:-1]):
        prev_time = int(inference_times[i + 1])
        with torch.no_grad():
            z_latent = z_latent_dict[t]
            x_latent = x_latent_dict[t]
            input_latent = torch.cat([z_latent, x_latent], dim=0)
            input_cond = torch.cat([con, con], dim=0)
            trg_indexs_list = [[1]]
            pixel_set = []
            noise_pred = call_unet(unet, input_latent, t,
                                   input_cond, trg_indexs_list, pixel_set)

        with torch.no_grad():
            mask_dict = controller.step_store
            controller.reset()
            # ------------------- 1. get mask ------------------- #
            layers = mask_dict.keys()
            mask_dict_by_res = {}
            map_list = []
            for layer in layers:
                mask = mask_dict[layer][0] # [8,1024]
                head, pix_num = mask.shape
                res = int(pix_num ** 0.5)
                if res == args.pixel_mask_res:
                    cross_maps = mask.reshape(head, res, res) # 8, 32,32
                    map_list.append(cross_maps)

            out = torch.cat(map_list, dim=0)   # [64,64]
            avg_attn = out.sum(0) / out.shape[0] # [64,64]
            mask = torch.where(avg_attn != 0 , 1, 0)
            image = mask.unsqueeze(-1).expand(*mask.shape, 4).cpu()  # res,res,3
            image = image.numpy().astype(np.uint8) * 255
            image = np.array(Image.fromarray(image).resize((64, 64))) / 255
            #pixel_mask = np.array(Image.fromarray(image).resize((512, 512)))
            mask_latent = torch.tensor(image).to(z_latent.device, dtype=z_latent.dtype)
            mask_latent = mask_latent.permute(2,0,1).unsqueeze(0)
            x_latent = x_latent * (1 -mask_latent) + z_latent * ( mask_latent)
            x_latent_dict[prev_time] = x_latent

            if prev_time == 0 :
                pil_img = Image.fromarray(latent2image(x_latent, vae, return_type='np'))
                pil_img.save(os.path.join(base_folder_dir, f'{name}_recon_{prev_time}.png'))
    return x_latent, time_steps, pil_images