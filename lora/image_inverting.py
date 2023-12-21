import argparse
from accelerate.utils import set_seed
import json
import os
import random
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

@torch.no_grad()
def customizing_latent2image(latents, vae, return_type='np'):
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
"""
def customizing_next_step(model_output: Union[torch.FloatTensor, np.ndarray],
                          timestep: int,
                          next_timestep: int,
                          sample: Union[torch.FloatTensor, np.ndarray],
                          alphas_cumprod_dict):
    alpha_prod_t = alphas_cumprod_dict[timestep]
    alpha_prod_t_next = alphas_cumprod_dict[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample
"""
def customizing_next_step(model_output: Union[torch.FloatTensor, np.ndarray],
                          alpha_prod_t,
                          alpha_prod_t_next,
                          sample: Union[torch.FloatTensor, np.ndarray],):
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
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

    #prev_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    prev_original_sample = (sample - beta_prod_t_matrix ** 0.5 * model_output) / alpha_prod_t_matrix ** 0.5
    #prev_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample_direction = (torch.ones_like(model_output) - alpha_prod_t_prev_matrix) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev_matrix ** 0.5 * prev_original_sample + prev_sample_direction
    return prev_sample

@torch.no_grad()
def latent2image_customizing(latents, vae, factor, return_type='np'):
    latents = factor * latents.detach()
    image = vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    return image

def latent2image_customizing_with_decoder(latents, student, factor, return_type='np'):
    latents = factor * latents.detach()
    image = student(latents)
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    return image

@torch.no_grad()
def ddim_loop(latent, context, inference_times, scheduler, unet, vae, student,
              base_folder_dir, attention_storer,alphas_cumprod_dict,
              vae_factor_dict):
    uncond_embeddings, cond_embeddings = context.chunk(2)
    time_steps = []
    latent = latent.clone().detach()
    latent_dict = {}
    noise_pred_dict = {}
    latent_dict[0] = latent
    pil_images = []
    with torch.no_grad():
        if args.customizing_decoder:
            if args.using_customizing_scheduling :
                factor = vae_factor_dict[0]
                np_img = latent2image_customizing_with_decoder(latent, student, factor, return_type='np')
            else :
                factor = 1/0.18215
                np_img = latent2image_customizing_with_decoder(latent, student, factor, return_type='np')
        else:
            np_img = latent2image(latent, vae, return_type='np')
    #pil_img = Image.fromarray(np_img)
    #pil_images.append(pil_img)
    #pil_img.save(os.path.join(base_folder_dir, f'original_sample.png'))
    flip_times = inference_times
    repeat_time = 0
    for i, t in enumerate(flip_times[:-1]):
        if repeat_time < args.repeat_time:
            next_time = flip_times[i + 1].item()
            latent_dict[int(t.item())] = latent
            time_steps.append(t.item())
            noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
            noise_pred_dict[int(t.item())] = noise_pred
            if args.with_new_noising_alphas_cumprod :
                latent = customizing_next_step(noise_pred,
                                               alphas_cumprod_dict[int(t.item())],
                                               alphas_cumprod_dict[int(next_time)],
                                               latent).to(noise_pred.dtype)
                latent = latent.to(vae.device)
            else :
                latent = next_step(noise_pred, int(t.item()), latent, scheduler)

            if args.customizing_decoder :
                if args.using_customizing_scheduling:
                    factor = vae_factor_dict[0]
                    np_img = latent2image_customizing_with_decoder(latent, student, factor, return_type='np')
                else:
                    factor = 1 / 0.18215
                    np_img = latent2image_customizing_with_decoder(latent, student, factor, return_type='np')
            else :
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
def recon_loop(latent_dict, context, inference_times, scheduler, unet, vae, student,
               base_folder_dir, vae_factor_dict, weight_dtype):
    uncon, con = context.chunk(2)
    if inference_times[0] < inference_times[1] :
        inference_times.reverse()
    #inference_times = inference_times[1:]
    latent = latent_dict[inference_times[0]]
    all_latent_dict = {}
    all_latent_dict[inference_times[0]] = latent
    time_steps = []
    pil_images = []
    with torch.no_grad():
        if args.customizing_decoder:
            factor = 1 / 0.18215
            np_img = latent2image_customizing_with_decoder(latent, student, factor, return_type='np')
            #np_img = latent2image(latent, vae, return_type='np')
        else :
            np_img = latent2image(latent, vae, return_type='np')
    pil_img = Image.fromarray(np_img)
    pil_images.append(pil_img)
    pil_img.save(os.path.join(base_folder_dir, f'recon_start_time_{inference_times[0]}.png'))
    latent_y = latent
    for i, t in enumerate(inference_times[:-1]):
        prev_time = int(inference_times[i + 1])
        time_steps.append(int(t))
        with torch.no_grad():
            noise_pred = call_unet(unet, latent, t, con, t, prev_time)
            latent = prev_step(noise_pred, int(t), latent, scheduler)
            if args.customizing_decoder :
                if args.using_customizing_scheduling:
                    factor = vae_factor_dict[0]
                    np_img = latent2image_customizing_with_decoder(latent, student, factor, return_type='np')
                else:
                    factor = 1 / 0.18215
                    np_img = latent2image_customizing_with_decoder(latent, student, factor, return_type='np')
            else :
                factor = 1/0.18215
                if args.customizing_decoder:
                    np_img = latent2image_customizing_with_decoder(latent, student, factor, return_type='np')
                else :
                    np_img = latent2image(latent, vae, return_type='np')
        #if prev_time == 0 :
        pil_img = Image.fromarray(np_img)
        pil_images.append(pil_img)
        pil_img.save(os.path.join(base_folder_dir, f'recon_{prev_time}.png'))
        latent = image2latent(np_img, vae, vae.device, weight_dtype)
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

    print(f'\n step 3. preparing accelerator')
    accelerator = train_util.prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

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
    device = accelerator.device
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

    print(f' (2.3) vae student model')
    print(f' (2.3.1) making encoder of vae')
    from diffusers import AutoencoderKL
    from STTraining import Student
    vae_encoder = vae.encoder
    vae_encoder_quantize = vae.quant_conv
    vae_encoder.requires_grad_(False)
    vae_encoder.eval()
    vae_encoder_quantize.requires_grad_(False)
    vae_encoder_quantize.eval()


    config_dict = vae.config
    student_vae = AutoencoderKL.from_config(config_dict)
    student_vae_decoder = student_vae.decoder
    student_vae_decoder_quantize = student_vae.post_quant_conv
    student = Student(student_vae_decoder, student_vae_decoder_quantize)
    print(f' (2.3.2) making decoder of vae')
    student_pretrained_dir = args.student_pretrained_dir
    model_state_dict = torch.load(student_pretrained_dir, map_location="cpu")
    state_dict = {}
    for k, v in model_state_dict.items():
        k_ = '.'.join(k.split('.')[1:])
        state_dict[k_] = v
    student.load_state_dict(state_dict, strict=True)
    student.requires_grad_(False)
    student.eval()

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
    #attention_storer = None
    register_attention_control(invers_unet, attention_storer)
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
    print(f' (2.3.4) apply trained state dict')
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
    network.to(device)
    vae_encoder.to(device, dtype=vae_dtype)
    vae_encoder_quantize.to(device, dtype=vae_dtype)
    student.to(device, dtype=vae_dtype)

    print(f' (2.4) scheduling factors')
    alphas_cumprod_dict = {}
    inference_decoding_factor = {}
    if args.using_customizing_scheduling :
        vae_factor_dict = '../result/decoder_factor.txt'
        with open(vae_factor_dict, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.strip()
            line = line.split(' : ')
            t, f = int(line[0]), float(line[1])
            inference_decoding_factor[t] = f


    elif args.with_new_noising_alphas_cumprod :
        alphas_cumprod_dict_dir = r'../result/inference_decoding_factor_txt.txt'
        with open(alphas_cumprod_dict_dir, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.strip()
            line = line.split(' : ')
            t, f = int(line[0]), float(line[1])
            alphas_cumprod_dict[t] = f
    print(f'inference_decoding_factor : {inference_decoding_factor}')



    print(f' \n step 3. ground-truth image preparing')
    print(f' (3.1) prompt condition')
    prompt = args.prompt
    invers_context = init_prompt(tokenizer, invers_text_encoder, device, prompt)
    context = init_prompt(tokenizer, text_encoder, device, prompt)

    print(f' (3.2) train images')
    """
    train_img_folder = os.path.join(args.concept_image_folder, 'train/good/rgb/30_rgb')
    train_images = os.listdir(train_img_folder)
    for i, train_img in enumerate(train_images) :
        if i < 1 :
            train_img_dir = os.path.join(train_img_folder, train_img)
            concept_name = train_img.split('.')[0]
            print(f' (2.3.1) inversion')
            image_gt_np = load_512(train_img_dir)
            latent = image2latent(image_gt_np, vae, device, weight_dtype)
            save_base_folder = os.path.join(output_dir,
                                            f'train/noising_inverse_unet_denoising_lora_{args.num_ddim_steps}_model_epoch_{model_epoch}')

            os.makedirs(save_base_folder, exist_ok=True)
            train_base_folder = os.path.join(save_base_folder, concept_name)
            os.makedirs(train_base_folder, exist_ok=True)
            inference_times = torch.cat([torch.tensor([999]), scheduler.timesteps, ], dim=0)
            flip_times = torch.flip(inference_times, dims=[0])  # [0,20, ..., 980]
            original_latent = latent.clone().detach()
            for ii, final_time in enumerate(flip_times[1:]):
                if final_time.item() == args.final_time :
                    timewise_save_base_folder = os.path.join(train_base_folder, f'final_time_{final_time.item()}')
                    print(f' - save_base_folder : {timewise_save_base_folder}')
                    os.makedirs(timewise_save_base_folder, exist_ok=True)
                    org_pil = Image.open(train_img_dir).resize((512, 512)).convert('RGB')

                    org_pil.save(os.path.join(timewise_save_base_folder, 'org.png'))
                    latent_dict, time_steps, pil_images = ddim_loop(latent=original_latent,
                                                                    context=invers_context,
                                                                    inference_times=flip_times[:ii + 2],
                                                                    scheduler=scheduler,
                                                                    unet=invers_unet,
                                                                    vae=vae,
                                                                    student = student,
                                                                    base_folder_dir=timewise_save_base_folder,
                                                                    attention_storer=attention_storer,
                                                                    alphas_cumprod_dict=alphas_cumprod_dict,
                                                                    vae_factor_dict=inference_decoding_factor)
                    attention_storer.reset()
                    time_steps.reverse()
                    print(f'time_steps : {time_steps}')
                    context = init_prompt(tokenizer, text_encoder, device, prompt)
                    print(f' (2.3.2) recon')
                    recon_latent_dict, _, _ = recon_loop(latent_dict=latent_dict,
                                                         context=context,
                                                         inference_times=time_steps,  # [20,0]
                                                         scheduler=scheduler,
                                                         unet=unet,
                                                         vae=vae,
                                                         student = student,
                                                         base_folder_dir=timewise_save_base_folder,
                                                         vae_factor_dict = inference_decoding_factor,
                                                         weight_dtype=weight_dtype)
                    attention_storer.reset()

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
        for j, test_img in enumerate(test_images):
            if j < 100 :
                train_img_dir = os.path.join(image_folder, test_img)
                mask_img_dir = os.path.join(mask_folder, test_img)
                concept_name = test_img.split('.')[0]
                print(f' (2.3.1) inversion')
                save_base_dir = os.path.join(class_base_folder, concept_name)
                os.makedirs(save_base_dir, exist_ok=True)
                image_gt_np = load_512(train_img_dir)
                latent = image2latent(image_gt_np, vae, device, weight_dtype)
                inference_times = torch.cat([torch.tensor([999]), scheduler.timesteps, ], dim=0)
                flip_times = torch.flip(inference_times, dims=[0])  # [0,20, ..., 980]
                original_latent = latent.clone().detach()
                for ii, final_time in enumerate(flip_times[1:]):
                    if final_time.item() == args.final_time:
                        timewise_save_base_folder = os.path.join(save_base_dir, f'final_time_{final_time.item()}')
                        print(f' - save_base_folder : {timewise_save_base_folder}')
                        os.makedirs(timewise_save_base_folder, exist_ok=True)
                        mask_pil = Image.open(mask_img_dir).resize((512, 512)).convert('RGB')
                        mask_pil.save(os.path.join(timewise_save_base_folder, 'mask.png'))
                        org_pil = Image.open(train_img_dir).resize((512, 512)).convert('RGB')
                        org_pil.save(os.path.join(timewise_save_base_folder, 'org.png'))
                        latent_dict, time_steps, pil_images = ddim_loop(latent=original_latent,
                                                                        context=invers_context,
                                                                        inference_times=flip_times[:ii + 2],
                                                                        scheduler=scheduler,
                                                                        unet=invers_unet,
                                                                        vae=vae,
                                                                        student=student,
                                                                        base_folder_dir=timewise_save_base_folder,
                                                                        attention_storer=attention_storer,
                                                                        alphas_cumprod_dict=alphas_cumprod_dict,
                                                                        vae_factor_dict=inference_decoding_factor)
                        attention_storer.reset()
                        time_steps.reverse()
                        print(f'time_steps : {time_steps}')
                        context = init_prompt(tokenizer, text_encoder, device, prompt)
                        print(f' (2.3.2) recon')
                        recon_latent_dict, _, _ = recon_loop(latent_dict=latent_dict,
                                                             context=context,
                                                             inference_times=time_steps,  # [20,0]
                                                             scheduler=scheduler,
                                                             unet=unet,
                                                             vae=vae,
                                                             student=student,
                                                             base_folder_dir=timewise_save_base_folder,
                                                             vae_factor_dict=inference_decoding_factor,
                                                             weight_dtype=weight_dtype)
                        attention_storer.reset()


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
    parser.add_argument("--inversion_experiment", action="store_true",)
    parser.add_argument("--repeat_time", type=int, default=1)
    parser.add_argument("--self_attn_threshold_time", type=int, default=1)
    parser.add_argument("--using_customizing_scheduling", action="store_true",)
    parser.add_argument("--final_time", type=int, default = 600)
    parser.add_argument("--customizing_decoder", action='store_true')
    parser.add_argument("--with_new_noising_alphas_cumprod", action='store_true')
    parser.add_argument("--student_pretrained_dir", type=str)
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    main(args)