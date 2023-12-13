import argparse
import json
import os
import random
from accelerate.utils import set_seed
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
import torch
from torch import nn
from attention_store import AttentionStore
from PIL import Image
import sys, importlib
from typing import Union
import numpy as np
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

def register_attention_control(unet : nn.Module, controller:AttentionStore) :
    """ Register cross attention layers to controller. """
    def ca_forward(self, layer_name):

        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)

            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if self.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                                                         device=query.device),
                                             query,key.transpose(-1, -2),beta=0,alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            if not is_cross_attention:
                # when self attention
                controller.self_query_key_value_caching(query_value=query.detach().cpu(),
                                                        key_value=key.detach().cpu(),
                                                        value_value=value.detach().cpu(),
                                                        layer_name=layer_name)
            if is_cross_attention :
                controller.cross_key_caching(key_value=query.detach().cpu(),
                                             layer_name=layer_name)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states
        return forward

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count

    cross_att_count = 0
    for net in unet.named_children():
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
    controller.num_att_layers = cross_att_count
def unregister_attention_control(unet : nn.Module, controller:AttentionStore) :
    """ Register cross attention layers to controller. """
    def ca_forward(self, layer_name):

        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            if not is_cross_attention and mask is not None:
                if args.self_key_control :
                    unkey, con_key = key.chunk(2)
                    key = torch.cat([unkey, mask[0][layer_name]], dim=0)
                unvalue, con_value = value.chunk(2)
                value = torch.cat([unvalue, mask[1][layer_name]], dim=0)

            if self.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                                                         device=query.device),
                                             query,key.transpose(-1, -2),beta=0,alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states
        return forward

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count

    cross_att_count = 0
    for net in unet.named_children():
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
    controller.num_att_layers = cross_att_count
def register_self_condition_giver(unet: nn.Module, collector, self_query_dict, self_key_dict,self_value_dict):

    def ca_forward(self, layer_name):
        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)

            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)


            if not is_cross_attention:
                if trg_indexs_list > args.self_attn_threshold_time or mask == 0 :
                    if hidden_states.shape[0] == 2 :
                        uncon_key, con_key = key.chunk(2)
                        uncon_value, con_value = value.chunk(2)
                        key = torch.cat([uncon_key,
                                         self_key_dict[trg_indexs_list][layer_name].to(query.device)], dim=0)
                        value = torch.cat([uncon_value,
                                           self_value_dict[trg_indexs_list][layer_name].to(query.device)], dim=0)
                    else :
                        key = self_key_dict[trg_indexs_list][layer_name].to(query.device)
                        value = self_value_dict[trg_indexs_list][layer_name].to(query.device)


            if self.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                                                         device=query.device),
                                             query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            if mask == 0 and not is_cross_attention :
                collector.store(attention_probs, layer_name)
            attention_probs = attention_probs.to(value.dtype)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states
        return forward

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count
    cross_att_count = 0
    for net in unet.named_children():
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
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
@torch.no_grad()
def image2latent(image, vae, device, weight_dtype):
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

@torch.no_grad()
def latent2image(latents, vae, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    return image

def init_prompt(tokenizer, text_encoder, device, prompt: str):
    uncond_input = tokenizer([""],
                             padding="max_length", max_length=tokenizer.model_max_length,
                             return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_input = tokenizer([prompt],
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt",)
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    return context

def call_unet(unet, noisy_latents, timesteps, text_conds, trg_indexs_list, mask_imgs):
    noise_pred = unet(noisy_latents, timesteps, text_conds,
                      trg_indexs_list=trg_indexs_list,
                      mask_imgs=mask_imgs, ).sample
    return noise_pred
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
    #next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_original_sample = (sample - beta_prod_t_matrix ** 0.5 * model_output) / alpha_prod_t_matrix ** 0.5
    #next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample_direction = (torch.ones_like(model_output) - alpha_prod_t_next_matrix) ** 0.5 * model_output

    #next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    next_sample = alpha_prod_t_next_matrix ** 0.5 * next_original_sample + next_sample_direction
    return next_sample

def prev_step(model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray],
              scheduler):
    timestep, prev_timestep = timestep, max( timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 0)
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
    beta_prod_t = 1 - alpha_prod_t
    
    prev_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    prev_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * prev_original_sample + prev_sample_direction
    return prev_sample

def customizing_prev_step(model_output: Union[torch.FloatTensor, np.ndarray],
                          sample: Union[torch.FloatTensor, np.ndarray],
                          alpha_dict,
                          timestep,
                          prev_timestep):
    alpha_prod_t = alpha_dict[timestep]
    alpha_prod_t_prev = alpha_dict[prev_timestep]
    beta_prod_t = 1 - alpha_prod_t
    prev_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    prev_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * prev_original_sample + prev_sample_direction
    return prev_sample


def inter_step(model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray],
              scheduler):
    timestep, prev_timestep = timestep, max( timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 0)
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
    a_t = (alpha_prod_t_prev/alpha_prod_t) ** 0.5
    beta_prod_t = 1 - alpha_prod_t
    b_t = -1 * (((beta_prod_t*alpha_prod_t_prev)/alpha_prod_t)**0.5) + (1-alpha_prod_t_prev)**0.5
    inter_sample = a_t * sample + b_t * model_output
    return inter_sample

@torch.no_grad()
def ddim_loop(latent, context, inference_times, scheduler, unet, vae, base_folder_dir, attention_storer):
    uncond_embeddings, cond_embeddings = context.chunk(2)
    time_steps = []
    latent = latent.clone().detach()
    latent_dict = {}
    noise_pred_dict = {}
    latent_dict[0] = latent
    pil_images = []
    with torch.no_grad():
        np_img = latent2image(latent, vae, return_type='np')
    pil_img = Image.fromarray(np_img)
    pil_images.append(pil_img)
    pil_img.save(os.path.join(base_folder_dir, f'original_sample.png'))
    #inference_times = torch.cat([torch.Tensor([999]), inference_times])
    #flip_times = torch.flip(inference_times, dims=[0])
    flip_times = inference_times
    repeat_time = 0
    for i, t in enumerate(flip_times[:-1]):
        if repeat_time < args.repeat_time:
            next_time = flip_times[i + 1].item()
            latent_dict[int(t.item())] = latent
            time_steps.append(t.item())
            # con_noise_pred = call_unet(unet, latent, t, cond_embeddings, None, None)
            # uncon_noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
            # if -1 only con, if 0, only uncon
            # noise_pred = uncon_noise_pred - args.inversion_weight * (con_noise_pred - uncon_noise_pred)
            noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
            noise_pred_dict[int(t.item())] = noise_pred
            latent = next_step(noise_pred, int(t.item()), latent, scheduler)
            with torch.no_grad():
                np_img = latent2image(latent, vae, return_type='np')
            pil_img = Image.fromarray(np_img)
            pil_images.append(pil_img)
            pil_img.save(os.path.join(base_folder_dir, f'noising_{next_time}.png'))
            repeat_time += 1
    time_steps.append(next_time)
    latent_dict[int(next_time)] = latent
    latent_dict_keys = latent_dict.keys()
    return latent_dict, time_steps, pil_images


@torch.no_grad()
def recon_loop(latent_dict, context, inference_times, scheduler, unet, vae, base_folder_dir, alpha_dict):
    latent = latent_dict[inference_times[0]]
    all_latent_dict = {}
    all_latent_dict[inference_times[0]] = latent
    time_steps = []
    pil_images = []
    with torch.no_grad():
        np_img = latent2image(latent, vae, return_type='np')
    pil_img = Image.fromarray(np_img)
    pil_images.append(pil_img)
    pil_img.save(os.path.join(base_folder_dir, f'recon_start_time_{inference_times[0]}.png'))
    latent_y = latent.clone().detach()
    for i, t in enumerate(inference_times[:-1]):

        prev_time = int(inference_times[i + 1])
        time_steps.append(int(t))
        """

        if args.classifier_free_guidance_infer :
            if t > args.cfg_check :
                input_latent = torch.cat([latent] * 2)
                trg_latent = latent_dict[prev_time]
                noise_pred = call_unet(unet, input_latent, t, context, t, prev_time)
                guidance_scales = [-80, -70,-60,-50,-40,-30,-20,-10,0, 1, 2, 3, 4, 5, 6, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
                latent_diff_dict = {}
                latent_dictionary = {}
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                for guidance_scale in guidance_scales:
                    inter_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latent_diff = torch.nn.functional.mse_loss(prev_step(inter_noise_pred, int(t), latent, scheduler).float(),
                                                               trg_latent.float(),
                                                               reduction='none')
                    latent_diff_dict[guidance_scale] = latent_diff.mean()
                    latent_dictionary[guidance_scale] = inter_noise_pred
                best_guidance_scale = sorted(latent_diff_dict.items(), key=lambda x : x[1].item())[0][0]
                noise_pred = latent_dictionary[best_guidance_scale]
                latent = prev_step(noise_pred, int(t), latent, scheduler)
            else :
                uncon, con = context.chunk(2)
                noise_pred = call_unet(unet, latent, t, con, t, prev_time)
                latent = prev_step(noise_pred, int(t), latent, scheduler)
        """
        if args.latent_coupling:
            trg_latent = latent_dict[prev_time]
            latent_loss_dict = {}
            latent_dictionary = {}
            uncon, con = context.chunk(2)
            noise_pred_y = call_unet(unet, latent_y,       t, con, t, prev_time)
            latent_x_inter = inter_step(noise_pred_y,int(t),latent, scheduler)
            noise_pred_x = call_unet(unet, latent_x_inter, t, con, t, prev_time)
            latent_y_inter =  inter_step(noise_pred_x,int(t),latent, scheduler)
            for p in [0.8, 0.85, 0.9, 0.95, 0.98]:
                latent_y = args.p * latent_y_inter + (1 - args.p) * latent_x_inter
                latent_x =   args.p * latent_x_inter + (1 - args.p) * latent_y_inter
                latent_loss = torch.nn.functional.mse_loss(latent_x,
                                                           trg_latent, reduction='none')
                latent_loss_dict[p] = latent_loss.mean()
                latent_dictionary[p] = latent_x
            best_p = sorted(latent_loss_dict.items(), key=lambda x : x[1].item())[0][0]
            latent = latent_dictionary[best_p]
            # trg_latent
        if args.using_customizing_scheduling :
            if args.classifier_free_guidance_infer :
                if t > args.cfg_check:
                    input_latent = torch.cat([latent] * 2)
                    trg_latent = latent_dict[prev_time]
                    noise_pred = call_unet(unet, input_latent, t, context, t, prev_time)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                    latent = customizing_prev_step(noise_pred,latent,alpha_dict, t, prev_time)
            else :
                uncon, con = context.chunk(2)
                noise_pred = call_unet(unet, latent, t, con, t, prev_time)
                latent = customizing_prev_step(noise_pred,latent,alpha_dict, t, prev_time)

        with torch.no_grad():
            np_img = latent2image(latent, vae, return_type='np')
        pil_img = Image.fromarray(np_img)
        pil_images.append(pil_img)
        #if prev_time == 0 :
        pil_img.save(os.path.join(base_folder_dir, f'recon_{prev_time}.png'))
        all_latent_dict[prev_time] = latent

    time_steps.append(prev_time)
    return all_latent_dict, time_steps, pil_images

def main(args) :

    print(f' \n step 1. setting')
    if args.process_title:
        setproctitle(args.process_title)
    else:
        setproctitle('parksooyeon')
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f" (1.2) save directory and save config")
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    print(f" (1.3) save dir")
    model_name = os.path.split(args.network_weights)[-1]
    model_name = os.path.splitext(model_name)[0]
    if 'last' not in model_name:
        model_epoch = int(model_name.replace('epoch-', ''))
    else:
        model_epoch = 'last'

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f'recon_check')
    os.makedirs(output_dir, exist_ok=True)

    print(f' \n step 2. make stable diffusion model')
    device = args.device
    print(f' (2.1) tokenizer')
    tokenizer = train_util.load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
    print(f' (2.2) SD')
    invers_text_encoder, vae, invers_unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype,device,
                                                                                                        unet_use_linear_projection_in_v2=False, )
    invers_text_encoders = invers_text_encoder if isinstance(invers_text_encoder, list) else [invers_text_encoder]
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
                              beta_end=SCHEDULER_LINEAR_END, beta_schedule=SCHEDLER_SCHEDULE,
                              rescale_betas_zero_snr=True,
                              )
    scheduler.set_timesteps(args.num_ddim_steps)
    inference_times = scheduler.timesteps

    print(f' (2.4) model to accelerator device')
    device = args.device
    if len(invers_text_encoders) > 1:
        invers_unet, invers_t_enc1, invers_t_enc2 = invers_unet.to(device), invers_text_encoders[0].to(device),invers_text_encoders[1].to(device)
        invers_text_encoder = [invers_t_enc1, invers_t_enc2]
        del invers_t_enc1, invers_t_enc2
        unet, t_enc1, t_enc2 = unet.to(device), text_encoders[0].to(device), text_encoders[1].to(device)
        text_encoder = [t_enc1, t_enc2]
        del t_enc1, t_enc2
    else:
        invers_unet, invers_text_encoder = invers_unet.to(device), invers_text_encoder.to(device)
        invers_text_encoders = [invers_text_encoder]
        unet, text_encoder = unet.to(device), text_encoder.to(device)
        text_encoders = [text_encoder]
    attention_storer = AttentionStore()
    register_attention_control(invers_unet, attention_storer)
    #register_attention_control(unet, attention_storer)
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

    print(f' \n step 3. ground-truth image preparing')
    print(f' (3.1) prompt condition')
    prompt = args.prompt
    invers_context = init_prompt(tokenizer, invers_text_encoder, device, prompt)
    context = init_prompt(tokenizer, text_encoder, device, prompt)

    print(f' (3.2) train images')
    train_img_folder = os.path.join(args.concept_image_folder, 'train/good/rgb')
    train_images = os.listdir(train_img_folder)
    for train_img in train_images :
        train_img_dir = os.path.join(train_img_folder, train_img)
        concept_name = train_img.split('.')[0]
        print(f' (2.3.1) inversion')
        image_gt_np = load_512(train_img_dir)
        latent = image2latent(image_gt_np, vae, device, weight_dtype)
        #if args.latent_coupling:
        #    save_base_folder = os.path.join(output_dir,f'train/inference_time_{args.num_ddim_steps}_model_epoch_{model_epoch}_latent_coupling_dynamic_p')
        #elif args.classifier_free_guidance_infer and args.using_customizing_scheduling:
        #    save_base_folder = os.path.join(output_dir,f'train/inference_time_{args.num_ddim_steps}_model_epoch_{model_epoch}_cfg_guidance_{args.cfg_check}_customizing_scheduling_lora_noising')
        #elif args.using_customizing_scheduling :
        #    save_base_folder = os.path.join(output_dir,f'train/inference_time_{args.num_ddim_steps}_model_epoch_{model_epoch}_customizing_scheduling')
        save_base_folder = os.path.join(output_dir, f'train/noising_scheduling_test')
        print(f'save_base_folder : {save_base_folder}')
        os.makedirs(save_base_folder, exist_ok=True)
        train_base_folder = os.path.join(save_base_folder, concept_name)
        os.makedirs(train_base_folder, exist_ok=True)
        # time_steps = 0,20,..., 980
        flip_times = torch.flip(torch.cat([torch.tensor([999]), inference_times, ], dim=0), dims=[0])  # [0,20, ..., 980, 999]
        original_latent = latent.clone().detach()
        final_time = flip_times[-1]
        timewise_save_base_folder = os.path.join(save_base_folder,f'final_time_{final_time.item()}')
        os.makedirs(timewise_save_base_folder, exist_ok=True)
        uncon, con = invers_context.chunk(2)
        noising_alphas_cumprod_dict = {}
        noising_alphas_cumprod_dict[0] = scheduler.alphas_cumprod[0]
        for i, present_t in enumerate(flip_times[:-1]):
            next_t = flip_times[i + 1]
            with torch.no_grad():
                noise_pred = call_unet(unet, latent, present_t, uncon, next_t, present_t)
                #target_latent = next_step(noise_pred,int(present_t.item()), latent, scheduler)
            alpha = scheduler.alphas_cumprod[present_t.item()].clone().detach()
            alpha.requires_grad = True
            optimizer = torch.optim.Adam([alpha], lr=0.01)
            alpha_prev = noising_alphas_cumprod_dict[present_t.item()]
            for j in range(10000):
                alpha_before = alpha.clone().detach()
                next_latent = ((alpha/alpha_prev)**0.5) * (latent - ((1-alpha_prev)**0.5)*noise_pred) + ((1-alpha)**0.5) * noise_pred
                noise_pred_inter = call_unet(unet, next_latent, next_t, uncon, next_t, present_t)
                #origin_latent = prev_step(noise_pred,int(next_t.item()),latent,scheduler)
                loss = torch.nn.functional.mse_loss(noise_pred_inter, noise_pred).mean()
                #loss = torch.nn.functional.mse_loss(next_latent, target_latent).mean()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if loss.item() < 0.000001 :
                    break
                if torch.isnan(alpha).any():
                    print('break cause nan')
                    alpha = alpha_before
                    break
            print(f'[new alpha] {next_t.item()} : {alpha.item()}')
            noising_alphas_cumprod_dict[next_t.item()] = alpha
            latent = next_latent
        break
        """        
            
                
        
        
        
        
        
        latent_dict, time_steps, pil_images = ddim_loop(latent=original_latent,
                                                        context=invers_context,
                                                        #context=context,
                                                        inference_times=flip_times,
                                                        scheduler=scheduler,
                                                        unet=invers_unet,
                                                        #unet=unet,
                                                        vae=vae,
                                                        base_folder_dir=timewise_save_base_folder,
                                                        attention_storer=attention_storer)
        # attention storer checking #
        layer_names = attention_storer.self_query_store.keys()
        self_query_dict, self_key_dict, self_value_dict = {}, {}, {}
        for layer in layer_names:
            self_query_list = attention_storer.self_query_store[layer]
            self_key_list = attention_storer.self_key_store[layer]
            self_value_list = attention_storer.self_value_store[layer]
            i = 1
            for self_query, self_key, self_value in zip(self_query_list, self_key_list, self_value_list):
                time_step = time_steps[i]
                if time_step not in self_query_dict.keys():
                    self_query_dict[time_step] = {}
                    self_query_dict[time_step][layer] = self_query
                else:
                    self_query_dict[time_step][layer] = self_query

                if time_step not in self_key_dict.keys():
                    self_key_dict[time_step] = {}
                    self_key_dict[time_step][layer] = self_key
                else:
                    self_key_dict[time_step][layer] = self_key

                if time_step not in self_value_dict.keys():
                    self_value_dict[time_step] = {}
                    self_value_dict[time_step][layer] = self_value
                else:
                    self_value_dict[time_step][layer] = self_value
                i += 1
        attention_storer.reset()

        context = init_prompt(tokenizer, text_encoder, device, prompt)
        time_steps.reverse()
        print(f' (2.3.2) customizing scheduling')
        latent = latent_dict[time_steps[0]]
        all_latent_dict = {}
        all_latent_dict[time_steps[0]] = latent
        pil_images = []
        with torch.no_grad():
            np_img = latent2image(latent, vae, return_type='np')
        pil_img = Image.fromarray(np_img)
        pil_images.append(pil_img)
        pil_img.save(os.path.join(timewise_save_base_folder, f'recon_start_time_{time_steps[0]}.png')) # 999

        inference_alpha_dict = {}
        inference_alpha_dict[time_steps[0]] = scheduler.alphas_cumprod[time_steps[0]]
        uncon, con = context.chunk(2)
        for i, t in enumerate(time_steps[:-1]):
            prev_time = int(time_steps[i + 1])
            trg_latent = latent_dict[prev_time]
            with torch.no_grad():
                noise_pred = call_unet(unet, latent, t, con, t, prev_time)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha = scheduler.alphas_cumprod[prev_time].clone().detach()
            alpha.requires_grad = True
            optimizer = torch.optim.Adam([alpha], lr=0.01)
            for i in range(1000) :
                beta_t = 1 - alpha_prod_t
                prev_original_sample = (latent - beta_t ** 0.5 * noise_pred) * (( alpha/ alpha_prod_t) ** 0.5)
                prev_sample_direction = (1 - alpha) ** 0.5 * noise_pred
                prev_sample = prev_original_sample + prev_sample_direction
                loss = torch.nn.functional.mse_loss(trg_latent.float(), prev_sample.float(), reduction='none')
                loss = loss.mean()
                if loss.item() < 0.00002 :
                    break
                else :
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print(f'prev_time : {prev_time}, alpha : {alpha}')
            if torch.isnan(alpha).any() :
                alpha = scheduler.alphas_cumprod[prev_time]
            inference_alpha_dict[prev_time] = alpha
        print(f' (2.3.3) reconstructing')
        # timesteps = [0,20]
        context = init_prompt(tokenizer, text_encoder, device, prompt)
        print(f' (2.3.2) recon')
        recon_latent_dict, _, _ = recon_loop(latent_dict=latent_dict,
                                             context=context,
                                             inference_times=time_steps,
                                             scheduler=scheduler,
                                             unet=unet,
                                             vae=vae,
                                             base_folder_dir=timewise_save_base_folder,
                                             alpha_dict=inference_alpha_dict,)
        break
        """
    """
    print(f' (3.2) test images')
    test_img_folder = os.path.join(args.concept_image_folder, 'test')
    test_base_folder = os.path.join(output_dir, 'test')
    os.makedirs(test_base_folder, exist_ok=True)
    classes = os.listdir(test_img_folder)
    for class_name in classes:
        class_folder = os.path.join(test_img_folder, class_name)
        class_base_folder = os.path.join(test_base_folder, class_name)
        os.makedirs(class_base_folder, exist_ok=True)

        image_folder = os.path.join(class_folder, 'rgb')
        mask_folder = os.path.join(class_folder, 'gt')
        test_images = os.listdir(image_folder)
        for test_img in test_images:
            test_img_dir = os.path.join(image_folder, test_img)
            mask_img_dir = os.path.join(mask_folder, test_img)
            mask_img_pil = Image.open(mask_img_dir)
            #concept_name = test_img.split('.')[0]
            if args.latent_coupling:
                save_base_folder = os.path.join(class_base_folder,
                                                f'inference_time_{args.num_ddim_steps}_model_epoch_{model_epoch}_latent_coupling_p_{args.p}')
            elif args.classifier_free_guidance_infer and args.using_customizing_scheduling :
                save_base_folder = os.path.join(class_base_folder,
                                                f'inference_time_{args.num_ddim_steps}_model_epoch_{model_epoch}_cfg_guidance_{args.cfg_check}_customizing_scheduling')
            #elif args.using_customizing_scheduling:
            #    save_base_folder = os.path.join(class_base_folder,
            #                                    f'inference_time_{args.num_ddim_steps}_model_epoch_{model_epoch}_customizing_scheduling')
            os.makedirs(save_base_folder, exist_ok=True)
            # inference_times = [980, 960, ..., 0]
            image_gt_np = load_512(test_img_dir)
            latent = image2latent(image_gt_np, vae, device, weight_dtype)
            flip_times = torch.flip(torch.cat([torch.tensor([999]), scheduler.timesteps, ], dim=0), dims=[0])  # [0,20, ..., 980]
            original_latent = latent.clone().detach()


            noising_alpha_dict = {}
            noising_alpha_dict[0] = scheduler.alphas_cumprod[0]
            uncond_embeddings, cond_embeddings = context.chunk(2)
            for ii, final_time in enumerate(flip_times[1:]):
                if final_time.item() == 999 :
                    latent = original_latent
                    timewise_save_base_folder = os.path.join(save_base_folder, f'final_time_{final_time.item()}')
                    os.makedirs(timewise_save_base_folder, exist_ok=True)
                    noising_steps = flip_times[:ii+2]
                    for i, t in enumerate(noising_steps[:-1]):

                        next_time = noising_steps[i + 1].item()
                        alpha = scheduler.alphas_cumprod[next_time].detach().clone()
                        alpha.requires_grad = True
                        optimizer = torch.optim.Adam([alpha], lr=0.01)
                        noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
                        next_latent = next_step(latent, noise_pred, alpha)
                        next_noise_pred = call_unet(unet, next_latent, next_time, uncond_embeddings, None, None)
                        alpha_prev = noising_alpha_dict[t.item()]
                        for j in range(10000) :
                            pred = ((alpha_prev / alpha)**0.5) * (next_latent - next_noise_pred * ((1-alpha)**0.5))
                            direction = ((1-alpha_prev)**0.5)*next_noise_pred
                            loss = torch.nn.functional.mse_loss((pred + direction).float(),next_latent).mean()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            if loss.item() < 0.0001 :
                                break
                        noising_alpha_dict[next_time.item()] = alpha
                        noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
                        latent





                    latent_dict, time_steps, pil_images = ddim_loop(latent=original_latent,
                                                                    context=invers_context,
                                                                    inference_times=flip_times[:ii + 2],
                                                                    scheduler=scheduler,
                                                                    unet=invers_unet,
                                                                    vae=vae,
                                                                    base_folder_dir=timewise_save_base_folder,
                                                                    attention_storer=attention_storer)

                    @torch.no_grad()
                    def ddim_loop(latent, context, inference_times, scheduler, unet, vae, base_folder_dir,
                                  attention_storer):
                        uncond_embeddings, cond_embeddings = context.chunk(2)
                        time_steps = []
                        latent = latent.clone().detach()
                        latent_dict = {}
                        noise_pred_dict = {}
                        latent_dict[0] = latent
                        pil_images = []
                        with torch.no_grad():
                            np_img = latent2image(latent, vae, return_type='np')
                        pil_img = Image.fromarray(np_img)
                        pil_images.append(pil_img)
                        pil_img.save(os.path.join(base_folder_dir, f'original_sample.png'))
                        # inference_times = torch.cat([torch.Tensor([999]), inference_times])
                        # flip_times = torch.flip(inference_times, dims=[0])
                        flip_times = inference_times
                        repeat_time = 0
                        for i, t in enumerate(flip_times[:-1]):
                            if repeat_time < args.repeat_time:
                                next_time = flip_times[i + 1].item()
                                latent_dict[int(t.item())] = latent
                                time_steps.append(t.item())
                                # con_noise_pred = call_unet(unet, latent, t, cond_embeddings, None, None)
                                # uncon_noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
                                # if -1 only con, if 0, only uncon
                                # noise_pred = uncon_noise_pred - args.inversion_weight * (con_noise_pred - uncon_noise_pred)
                                noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
                                noise_pred_dict[int(t.item())] = noise_pred
                                latent = next_step(noise_pred, int(t.item()), latent, scheduler)
                                with torch.no_grad():
                                    np_img = latent2image(latent, vae, return_type='np')
                                pil_img = Image.fromarray(np_img)
                                pil_images.append(pil_img)
                                pil_img.save(os.path.join(base_folder_dir, f'noising_{next_time}.png'))
                                repeat_time += 1
                        time_steps.append(next_time)
                        latent_dict[int(next_time)] = latent
                        latent_dict_keys = latent_dict.keys()
                        print(f'time_Steps : {time_steps}')
                        return latent_dict, time_steps, pil_images




                    context = init_prompt(tokenizer, text_encoder, device, prompt)
                    time_steps.reverse()
                    print(f' (2.3.2) customizing scheduling')
                    latent = latent_dict[time_steps[0]]
                    all_latent_dict = {}
                    all_latent_dict[time_steps[0]] = latent
                    pil_images = []
                    with torch.no_grad():
                        np_img = latent2image(latent, vae, return_type='np')
                    pil_img = Image.fromarray(np_img)
                    pil_images.append(pil_img)
                    pil_img.save(
                        os.path.join(timewise_save_base_folder, f'recon_start_time_{time_steps[0]}.png'))  # 999

                    inference_alpha_dict = {}
                    inference_alpha_dict[time_steps[0]] = scheduler.alphas_cumprod[time_steps[0]]
                    uncon, con = context.chunk(2)
                    print(f'alpha dict of {int(time_steps[0])}')
                    inference_alpha_dict[int(time_steps[0])] = scheduler.alphas_cumprod[int(time_steps[0])]
                    for i, t in enumerate(time_steps[:-1]):
                        prev_time = int(time_steps[i + 1])
                        trg_latent = latent_dict[prev_time]
                        with torch.no_grad():
                            noise_pred = call_unet(invers_unet, latent, t, invers_context.chunk(2)[0], t, prev_time)
                        alpha_prod_t = inference_alpha_dict[t]
                        alpha = scheduler.alphas_cumprod[prev_time].clone().detach()
                        alpha.requires_grad = True
                        optimizer = torch.optim.Adam([alpha], lr=0.01)
                        for i in range(5000):
                            beta_t = 1 - alpha_prod_t
                            prev_original_sample = latent * ((alpha/alpha_prod_t)**0.5)
                            a = (1 - alpha) ** 0.5 * noise_pred
                            b = ((alpha * beta_t / alpha_prod_t) ** 0.5) * noise_pred
                            prev_sample_direction = a-b
                            prev_sample = prev_original_sample + prev_sample_direction
                            loss = torch.nn.functional.mse_loss(trg_latent.float(), prev_sample.float(),
                                                                reduction='none')
                            loss = loss.mean()
                            if loss.item() < 0.00002:
                                break
                            else:
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                        print(f'prev_time : {prev_time}, alpha : {alpha}')
                        if torch.isnan(alpha).any():
                            alpha = scheduler.alphas_cumprod[prev_time]
                        inference_alpha_dict[prev_time] = alpha
                        print(f'alpha dict of {prev_time}')
                    print(f' (2.3.3) reconstructing')
                    # timesteps = [0,20]
                    context = init_prompt(tokenizer, text_encoder, device, prompt)
                    print(f' (2.3.2) recon')
                    recon_latent_dict, _, _ = recon_loop(latent_dict=latent_dict,
                                                         context=invers_context,
                                                         inference_times=time_steps,
                                                         scheduler=scheduler,
                                                         #unet=unet,
                                                         unet=invers_unet,
                                                         vae=vae,
                                                         base_folder_dir=timewise_save_base_folder,
                                                         alpha_dict=inference_alpha_dict, )
            break
        break
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    # step 2. model

    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う", )
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
    parser.add_argument("--concept_image", type=str,
                        default = '/data7/sooyeon/MyData/perfusion_dataset/td_100/100_td/td_1.jpg')
    parser.add_argument("--mask_image_folder", type=str,)
    parser.add_argument("--prompt", type=str,
                        default = 'teddy bear, wearing like a super hero')
    parser.add_argument("--negative_prompt", type=str,
                        default = 'low quality, worst quality, bad anatomy,bad composition, poor, low effort')
    parser.add_argument("--concept_image_folder", type=str)
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--self_key_control", action='store_true')
    parser.add_argument("--self_attn_threshold_time", type=int, default = 900)
    parser.add_argument("--inversion_experiment", action="store_true",)
    parser.add_argument("--repeat_time", type=int, default=1)
    parser.add_argument("--latent_coupling", action="store_true",)
    parser.add_argument("--classifier_free_guidance_infer", action="store_true", )
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--using_customizing_scheduling", action="store_true", )

    parser.add_argument("--cfg_check", type=int, default=200)
    parser.add_argument("--inversion_weight", type=float, default=3.0)
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    main(args)
