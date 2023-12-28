import torch
import numpy as np
from typing import Union
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_utils import call_unet
from image_utils import latent2image
from PIL import Image

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



@torch.no_grad()
def ddim_loop(args, latent, context, inference_times, scheduler, unet, vae, final_time, base_folder_dir, name,):
    if context.shape[0] == 1:
        cond_embeddings = context
    else :
        uncond_embeddings, cond_embeddings = context.chunk(2)
    time_steps = []
    latent = latent.clone().detach()
    latent_dict = {}
    noise_pred_dict = {}
    latent_dict[0] = latent
    pil_images = []
    flip_times = inference_times
    for i, t in enumerate(flip_times[:-1]):
        next_time = flip_times[i + 1]
        if next_time <= final_time :
            latent_dict[int(t)] = latent
            time_steps.append(t)
            noise_pred = call_unet(unet, latent, t, cond_embeddings, None, None)
            noise_pred_dict[int(t)] = noise_pred
            latent = next_step(noise_pred, int(t), latent, scheduler)
            np_img = latent2image(latent, vae, return_type='np')
            pil_img = Image.fromarray(np_img)
            pil_images.append(pil_img)
            if next_time == final_time :
                pil_img.save(os.path.join(base_folder_dir, f'{name}_noising_{next_time}.png'))
    time_steps.append(final_time)
    latent_dict[int(final_time)] = latent
    return latent_dict, time_steps, pil_images



@torch.no_grad()
def recon_loop(args, latent_dict, start_latent, context, inference_times, scheduler, unet, vae, base_folder_dir,
               controller, name):
    if context.shape[0] == 2:
        uncon, con = context.chunk(2)
    else:
        con = context
    # inference_times = [100,80, ... 0]
    latent = start_latent
    all_latent_dict = {}

    print(f'inference_times : {inference_times}')

    all_latent_dict[inference_times[0]] = latent
    time_steps = []
    pil_images = []
    with torch.no_grad():
        np_img = latent2image(latent, vae, return_type='np')
    pil_img = Image.fromarray(np_img)
    pil_images.append(pil_img)
    #pil_img.save(os.path.join(base_folder_dir, f'{name}_only_infer_recon_start_time_{inference_times[0]}.png'))
    for i, t in enumerate(inference_times[:-1]):
        if latent_dict is not None:
            z_latent = latent_dict[inference_times[i]]
        x_latent = latent
        prev_time = int(inference_times[i + 1])
        time_steps.append(int(t))
        with torch.no_grad():
            if latent_dict is not None:
                input_latent = torch.cat([z_latent, x_latent], dim=0)
                input_cond = torch.cat([con, con], dim=0)
                trg_indexs_list = [[1]]
                pixel_set = []
            else :
                input_latent = x_latent
                input_cond = con
                trg_indexs_list = None
                pixel_set = None

            noise_pred = call_unet(unet,
                                   input_latent,
                                   t,
                                   input_cond,
                                   trg_indexs_list,
                                   pixel_set)

            if latent_dict is not None:

                mask_dict = controller.step_store
                controller.reset()
                layers = mask_dict.keys()
                mask_dict_by_res = {}
                for layer in layers:
                    mask = mask_dict[layer] # object positioned mask
                    mask = mask[0] # [8,1024]
                    head, pix_num = mask.shape
                    res = int(pix_num ** 0.5)
                    if res not in mask_dict_by_res.keys() :
                        mask_dict_by_res[res] = []
                    cross_maps = mask.reshape(head, res, res) # 8, 32,32
                    mask_dict_by_res[res].append(cross_maps)
                mask_res_dict = {}
                for resolution in mask_dict_by_res.keys():
                    if resolution == args.pixel_mask_res :
                        map_list = mask_dict_by_res[resolution]
                        out = torch.cat(map_list, dim=0)  # [num, 64,64]
                        avg_attn = out.sum(0) / out.shape[0]
                        mask_res_dict[resolution] = avg_attn

                images = []
                for res in mask_res_dict.keys() :
                    image = mask_res_dict[res]
                    image = 255 * image / image.max()
                    image = image.unsqueeze(-1).expand(*image.shape, 4).cpu()  # res,res,3
                    image = image.numpy().astype(np.uint8)
                    image = np.array(Image.fromarray(image).resize((64,64)))
                #print(f'resolution {args.pixel_mask_res}, mask : {image.shape}')
                #mask_latent = torch.where(mask_latent> 0, 1, 0) # this means all mask_lants is bigger than 0
                mask_latent = torch.tensor(image).to(z_latent.device, dtype = z_latent.dtype)
                mask_latent = mask_latent.permute(2,0,1).unsqueeze(0)/255
                print(f'mask_latent : {mask_latent}')
                #z_noise_pred, y_noise_pred = noise_pred.chunk(2)
                y_latent = z_latent + (1-mask_latent) + x_latent * (mask_latent)
                y_noise_pred = call_unet(unet,y_latent,t,con, None, None)
                y_latent = prev_step(y_noise_pred, int(t), y_latent, scheduler)
            else :
                y_latent = prev_step(noise_pred, t, x_latent, scheduler)

            # --------------------- mask --------------------- #
            latent = y_latent
            controller.reset()
            np_img = latent2image(latent, vae, return_type='np')
            pil_img = Image.fromarray(np_img)
            pil_images.append(pil_img)
            pil_img.save(os.path.join(base_folder_dir, f'{name}_recon_{prev_time}.png'))
        all_latent_dict[prev_time] = latent
    return all_latent_dict, time_steps, pil_images
