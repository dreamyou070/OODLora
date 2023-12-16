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
import copy

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
from diffusers import (DDPMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,
                       DPMSolverSinglestepScheduler,
                       LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler,
                       KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers import DDIMScheduler


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
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
def latent2image_customizing(latents, vae, factor, return_type='np'):
    latents = factor * latents.detach()
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
                           return_tensors="pt", )
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
    timestep, next_timestep = timestep, min(
        timestep + scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999)
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_matrix = torch.ones_like(model_output) * alpha_prod_t
    alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
    alpha_prod_t_next_matrix = torch.ones_like(model_output) * alpha_prod_t_next
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_matrix = torch.ones_like(model_output) * beta_prod_t
    # next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_original_sample = (sample - beta_prod_t_matrix ** 0.5 * model_output) / alpha_prod_t_matrix ** 0.5
    # next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample_direction = (torch.ones_like(model_output) - alpha_prod_t_next_matrix) ** 0.5 * model_output

    # next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    next_sample = alpha_prod_t_next_matrix ** 0.5 * next_original_sample + next_sample_direction
    return next_sample

def customizing_next_step(model_output: Union[torch.FloatTensor, np.ndarray],
                          alpha_cumprod_t, alpha_cumprod_t_next,
                          sample: Union[torch.FloatTensor, np.ndarray],):
    sample_coefficient = (alpha_cumprod_t_next/alpha_cumprod_t)**0.5
    model_output_coefficient = sample_coefficient * ((1-alpha_cumprod_t)**0.5) - (1-alpha_cumprod_t_next)**0.5
    next_sample = sample_coefficient * sample - model_output_coefficient * model_output
    return next_sample

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
            #if mask == 0 and not is_cross_attention :
            #    collector.store(attention_probs, layer_name)
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
def prev_step(model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray],
              scheduler):
    timestep, prev_timestep = timestep, max(
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 0)
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod

    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
    beta_prod_t = 1 - alpha_prod_t

    prev_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    prev_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * prev_original_sample + prev_sample_direction
    return prev_sample


@torch.no_grad()
def ddim_loop(latent, context, inference_times, scheduler, unet, vae, base_folder_dir):
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
    flip_times = inference_times
    repeat_time = 0
    for i, t in enumerate(flip_times[:-1]):
        next_time = flip_times[i + 1].item()
        latent_dict[int(t.item())] = latent
        time_steps.append(t.item())
        noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
        noise_pred_dict[int(t.item())] = noise_pred
        latent = next_step(noise_pred, int(t.item()), latent, scheduler)
        with torch.no_grad():
            np_img = latent2image(latent, vae, return_type='np')
            pil_img = Image.fromarray(np_img)
            pil_images.append(pil_img)
            pil_img.save(os.path.join(base_folder_dir, f'noising_{next_time}.png'))
    time_steps.append(next_time)
    latent_dict[int(next_time)] = latent
    return latent_dict, time_steps, pil_images

@torch.no_grad()
def customizing_ddim_loop(latent, context, inference_times, scheduler, unet, vae, base_folder_dir, customizing_alphas_cumprod_dict):
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
    flip_times = inference_times
    repeat_time = 0
    for i, t in enumerate(flip_times[:-1]):
        next_time = flip_times[i + 1].item()
        latent_dict[int(t.item())] = latent
        time_steps.append(t.item())
        noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
        latent = customizing_next_step(noise_pred,
                                       alpha_cumprod_t = customizing_alphas_cumprod_dict[int(t.item())],
                                       alpha_cumprod_t_next = customizing_alphas_cumprod_dict[int(next_time)],
                                       sample = latent)
        with torch.no_grad():
            np_img = latent2image(latent, vae, return_type='np')
            pil_img = Image.fromarray(np_img)
            pil_images.append(pil_img)
            pil_img.save(os.path.join(base_folder_dir, f'noising_{next_time}.png'))
    time_steps.append(next_time)
    latent_dict[int(next_time)] = latent
    return latent_dict, time_steps, pil_images

@torch.no_grad()
def recon_loop(latent_dict, context, inference_times, scheduler, unet, vae, base_folder_dir, vae_factor_dict):
    uncon, con = context.chunk(2)
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
    for i, t in enumerate(inference_times[:-1]):
        prev_time = int(inference_times[i + 1])
        time_steps.append(int(t))
        with torch.no_grad():
            noise_pred = call_unet(unet, latent, t, uncon, t, prev_time)
            latent = prev_step(noise_pred, int(t), latent, scheduler)
            if vae_factor_dict :
                factor = float(vae_factor_dict[prev_time])
            else :
                factor =  1 / 0.18215
            np_img = latent2image_customizing(latent, vae, factor, return_type='np')
        pil_img = Image.fromarray(np_img)
        pil_images.append(pil_img)
        pil_img.save(os.path.join(base_folder_dir, f'recon_{prev_time}.png'))
        all_latent_dict[prev_time] = latent
    time_steps.append(prev_time)
    return all_latent_dict, time_steps, pil_images

def main(args):

    print(f' ------- finding best scheduler for noising -------')

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
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f' \n step 2. make stable diffusion model')
    device = args.device
    print(f' (2.1) tokenizer')
    tokenizer = train_util.load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
    print(f' (2.2) SD')
    invers_text_encoder, vae, invers_unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype, device,
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
    elif args.sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif args.sample_sampler == "lms" or args.sample_sampler == "k_lms":
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
    else:
        scheduler_cls = DDIMScheduler
    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction "

    SCHEDULER_LINEAR_START = 0.00085
    SCHEDULER_LINEAR_END = 0.0120
    SCHEDULER_TIMESTEPS = 1000
    SCHEDLER_SCHEDULE = "scaled_linear"
    scheduler = scheduler_cls(num_train_timesteps=SCHEDULER_TIMESTEPS, beta_start=SCHEDULER_LINEAR_START,
                              beta_end=SCHEDULER_LINEAR_END, beta_schedule=SCHEDLER_SCHEDULE, )
    scheduler.set_timesteps(args.num_ddim_steps)
    inference_times = scheduler.timesteps

    print(f' (2.4) model to accelerator device')
    device = args.device
    if len(invers_text_encoders) > 1:
        invers_unet, invers_t_enc1, invers_t_enc2 = invers_unet.to(device), invers_text_encoders[0].to(device), \
        invers_text_encoders[1].to(device)
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

    print(f' (2.5) network')
    sys.path.append(os.path.dirname(__file__))
    network_module = importlib.import_module(args.network_module)

    print(f' (2.5.1) merging weights')
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value

    print(f' (2.5.2) make network')
    if args.dim_from_weights:
        network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet,
                                                                **net_kwargs)
    else:
        network = network_module.create_network(1.0,
                                                args.network_dim,
                                                args.network_alpha,
                                                vae, text_encoder, unet, neuron_dropout=args.network_dropout,
                                                **net_kwargs, )

    print(f' (2.5.3) apply trained state dict')
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        print(f'Loading Network Weights')
        info = network.load_weights(args.network_weights)
    network.to(device)


    print(f' \n step 3. ground-truth image preparing')
    print(f' (3.1) prompt condition')
    prompt = args.prompt
    invers_context = init_prompt(tokenizer, invers_text_encoder, device, prompt)


    print(f' (3.2) train images')
    train_img_folder = os.path.join(args.concept_image_folder, 'train/good/rgb')
    train_images = os.listdir(train_img_folder)
    noising_alphas_cumprod_text_file = r'../result/lora_noising_scheduler_alphas_cumprod.txt'
    customizing_alphas_cumprod_dict = {}
    customizing_alphas_cumprod_dict[0] = scheduler.alphas_cumprod[0]
    line = f'0 : {scheduler.alphas_cumprod[0].clone().detach().item()}'
    with open(noising_alphas_cumprod_text_file, 'a') as ff:
        ff.write(line + '\n')
    for train_img in train_images:
        train_img_dir = os.path.join(train_img_folder, train_img)
        print(f' (2.3.1) get suber image')
        image_gt_np = load_512(train_img_dir)
        latent = image2latent(image_gt_np, vae, device, weight_dtype)
        parent, network_name = os.path.split(args.network_weights)
        name, ext = os.path.splitext(network_name)
        save_base_folder = os.path.join(parent, f'inference_scheduling')
        os.makedirs(save_base_folder, exist_ok=True)
        flip_times = torch.flip(torch.cat([torch.tensor([999]), inference_times, ], dim=0), dims=[0])  # [0,20, ..., 980, 999]
        uncon, con = invers_context.chunk(2)
        print(f' (2.3.2) inversing')
        vae.eval()
        vae.requires_grad_(False)
        for i, present_t in enumerate(flip_times[:-1]):
            next_t = flip_times[i + 1] # torch
            with torch.no_grad():
                noise_pred = call_unet(unet, latent, present_t, uncon, next_t, present_t)

            alpha_cumprod_t = customizing_alphas_cumprod_dict[present_t.item()]
            alpha = scheduler.alphas_cumprod[next_t.item()].detach()
            print(f'alpha : {alpha} | type of alpha : {type(alpha)}')
            alpha.requires_grad = True
            optimizer = torch.optim.Adam([alpha], lr=0.001)

            for j in range(10000):
                alpha_before = alpha.clone().detach()
                latent_next = customizing_next_step(noise_pred, alpha_cumprod_t, alpha, latent)
                next_noise_pred = call_unet(unet, latent_next, next_t, uncon, next_t, present_t)
                recon_latent = prev_step(next_noise_pred, next_t.item(), sample = latent_next, scheduler=scheduler)
                loss = torch.nn.functional.mse_loss(latent.float(), recon_latent.float(), reduction="none")
                loss = loss.mean([1,2,3]).mean()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                print(f'loss : {loss.item()}')
                if loss.item() < 0.00005 :
                    break
                if torch.isnan(alpha).any():
                    alpha = alpha_before
                    break
            customizing_alphas_cumprod_dict[next_t.item()] = alpha.clone().detach().item()
            latent = latent_next
            # ----------------------------------------------------------------------------------------------- #
            # Testing
            #np_img = latent2image(recon_latent , vae, return_type='np')
            #pil_img = Image.fromarray(np_img)
            #pil_img.save(os.path.join(save_base_folder, f'recon_{int(present_t.item())}.png'))  # 999
            line = f'{next_t.item()} : {alpha.clone().detach()}'
            with open(noising_alphas_cumprod_text_file, 'a') as ff:
                ff.write(line + '\n')
        break

    with open(noising_alphas_cumprod_text_file, 'r') as f:
        content = f.readlines()
    noising_alphas_cumprod_dict = {}
    for line in content:
        line = line.strip()
        line = line.split(' : ')
        t, f = int(line[0]), float(line[1])
        noising_alphas_cumprod_dict[t] = f

    print(f' (3.3) random check')
    for train_img in train_images:
        concept = train_img.split('.')[0]
        train_img_dir = os.path.join(train_img_folder, train_img)
        print(f' (3.3.1) get suber image')
        image_gt_np = load_512(train_img_dir)
        latent = image2latent(image_gt_np, vae, device, weight_dtype)
        base_folder = os.path.join(save_base_folder, f'inference_scheduling')
        os.makedirs(base_folder, exist_ok=True)
        image_folder = os.path.join(base_folder, f'train_image_{concept}')
        os.makedirs(image_folder, exist_ok=True)
        flip_times = torch.flip(torch.cat([torch.tensor([999]), inference_times, ], dim=0), dims=[0])  # [0,20, ..., 980, 999]
        final_time = flip_times[-1]
        original_latent = latent.clone().detach()

        print(f' (2.3.2) inversing')
        latent_dict, time_steps, pil_images = customizing_ddim_loop(latent=original_latent,
                                                                    context=invers_context,
                                                                    inference_times = flip_times,
                                                                    scheduler=scheduler,
                                                                    unet=unet,
                                                                    vae=vae,
                                                                    base_folder_dir=image_folder,
                                                                    noising_alphas_cumprod_dict=noising_alphas_cumprod_dict)
        time_steps.reverse()
        recon_loop(latent_dict=latent_dict,
                   context=invers_context,
                   inference_times=time_steps,
                   scheduler=scheduler,
                   unet=unet,
                   vae=vae,
                   base_folder_dir=image_folder,
                   vae_factor_dict = None)

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
                        default='/data7/sooyeon/MyData/perfusion_dataset/td_100/100_td/td_1.jpg')
    parser.add_argument("--mask_image_folder", type=str, )
    parser.add_argument("--prompt", type=str,
                        default='teddy bear, wearing like a super hero')
    parser.add_argument("--negative_prompt", type=str,
                        default='low quality, worst quality, bad anatomy,bad composition, poor, low effort')
    parser.add_argument("--concept_image_folder", type=str)
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--self_attn_threshold_time", type=int, default=1)
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    main(args)
