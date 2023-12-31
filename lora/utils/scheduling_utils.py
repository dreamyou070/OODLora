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
    # inf_time : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
    # 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450,
    # 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680,
    # 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910,
    # 920, 930, 940, 950, 960, 970, 980, 990]
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
            print(f'final_time : {final_time}')
            time_steps.append(final_time)
            latent_dict[int(final_time)] = latent
            break
    if final_time not in latent_dict.keys():
        latent_dict[int(final_time)] = latent

    return latent_dict, time_steps, pil_images



@torch.no_grad()
def recon_loop(args, z_latent_dict, start_latent, gt_pil, context, inference_times, scheduler, unet, vae, base_folder_dir, controller, name,weight_dtype):
    original_latent = z_latent_dict[0]

    if context.shape[0] == 2:
        good_bad_con, good_con = context.chunk(2)
    else:
        good_con = context

    noise_pred = call_unet(unet, original_latent, 0, good_bad_con, [[1]], None)
    map_dict = controller.step_store
    controller.reset()
    cls_score_list, good_score_list, bad_score_list = [], [], []

    mask_dict = {}

    for layer in map_dict.keys():
        position_map = map_dict[layer][1]
        print(f'type of position_map : {type(position_map)}')
        mask_dict[layer] = position_map
        scores = map_dict[layer][0]
        cls_score, good_score = scores.chunk(2, dim=-1)
        # head, pix_num, 1
        cls_score_list.append(cls_score)
        good_score_list.append(good_score)
    cls_score = torch.cat(cls_score_list, dim=0).float().mean(dim=0).squeeze().reshape(int(args.cross_map_res[0]), int(args.cross_map_res[0]))  # [res*res]
    good_score = torch.cat(good_score_list, dim=0).float().mean(dim=0).squeeze().reshape(int(args.cross_map_res[0]), int(args.cross_map_res[0]))  # [res*res
    mask_latent = torch.where(cls_score < good_score + 0.1 , 1, 0)  # [16,16]
    print(f'cls_score : {cls_score}')
    print(f'good_score : {good_score}')
    print(f'mask latent : {mask_latent}')
    import time
    time.sleep(100)
    mask_img = mask_latent.cpu().numpy().astype(np.uint8)  # 1 means bad position
    mask_img = np.array(Image.fromarray(mask_img).resize((64, 64)))
    mask_latent = torch.tensor(mask_img).unsqueeze(0).unsqueeze(0).to(original_latent.device,
                                                                      dtype=original_latent.dtype)
    Image.fromarray(mask_img * 255).save(os.path.join(base_folder_dir, f'predicted_mask.png'))



    # inference_times = [100,80, ... 0]
    x_latent = start_latent
    x_latent_dict = {}
    x_latent_dict[inference_times[0]] = x_latent
    for i, t in enumerate(inference_times[:-1]):
        prev_time = int(inference_times[i + 1])
        with torch.no_grad():
            for i in range(args.inner_iteration) :

                z_latent = z_latent_dict[t]
                x_latent = x_latent_dict[t]
                input_latent = torch.cat([z_latent, x_latent], dim=0)
                input_cond = torch.cat([good_con, good_con], dim=0)
                trg_indexs_list = [[1]]
                noise_pred = call_unet(unet, input_latent, t, input_cond, trg_indexs_list, mask_dict)
                x_latent = x_latent * (1 - mask_latent) + z_latent * (mask_latent)
                #x_latent_dict[t] = x_latent
            x_noise_pred = call_unet(unet, x_latent, t, good_con, None, None)
            #z_noise_pred, x_noise_pred = noise_pred.chunk(2)
            x_latent = prev_step(x_noise_pred, t, x_latent, scheduler)
            x_latent_dict[prev_time] = x_latent
            pil_img = Image.fromarray(latent2image(x_latent, vae, return_type='np'))
            pil_img.save(os.path.join(base_folder_dir, f'{name}_recon_{t}.png'))
    pil_img = Image.fromarray(latent2image(x_latent, vae, return_type='np'))
    pil_img.save(os.path.join(base_folder_dir, f'{name}_recon_{prev_time}.png'))