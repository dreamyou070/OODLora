import argparse
import os
import random
import json
from accelerate.utils import set_seed
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
import torch
from torch import nn
from attention_store import AttentionStore
import wandb
import numpy as np
from PIL import Image
import sys, importlib
from typing import Union
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
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
                       LMSDiscreteScheduler,PNDMScheduler,DDIMScheduler,EulerDiscreteScheduler,HeunDiscreteScheduler,
                       KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler)


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
            """
            else :
                query, key, value = controller.cross_query_key_value_caching(query_value=query,
                                                                             key_value=key,
                                                                             value_value=value,
                                                                             layer_name=layer_name)
            """
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

def register_self_condition_giver(unet: nn.Module, self_query_dict, self_key_dict,self_value_dict):

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
                query = self_query_dict[trg_indexs_list][layer_name].to(query.device)
                key = self_key_dict[trg_indexs_list][layer_name].to(query.device)
                value = self_value_dict[trg_indexs_list][layer_name].to(query.device)

            if self.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                            device=query.device),
                query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
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
    alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
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
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
    beta_prod_t = 1 - alpha_prod_t
    prev_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    prev_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * prev_original_sample + prev_sample_direction
    return prev_sample
@torch.no_grad()
def ddim_loop(latent, context, inference_times, scheduler, unet, vae, base_folder_dir):
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    time_steps = []
    latent = latent.clone().detach()
    latent_dict = {}
    pil_images = []
    for t in torch.flip(inference_times, dims=[0]):
        latent_dict[t.item()] = latent
        with torch.no_grad():
            np_img = latent2image(latent, vae, return_type='np')
        pil_img = Image.fromarray(np_img)
        pil_images.append(pil_img)
        pil_img.save(os.path.join(base_folder_dir, f'with_con_with_self_qkv_inversion_{t.item()}.png'))
        # ----------------------------------------------------------------------------
        time_steps.append(t.item())
        noise_pred = call_unet(unet, latent, t, cond_embeddings, None, None)
        latent = next_step(noise_pred, t.item(), latent, scheduler)
        all_latent.append(latent)
    return all_latent, time_steps, pil_images

@torch.no_grad()
def recon_loop(latent,context,inference_times,scheduler, unet, vae,
               self_query_dict, self_key_dict,self_value_dict, base_folder_dir) :
    register_self_condition_giver(unet, self_query_dict, self_key_dict,self_value_dict)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    time_steps = []
    latent_dict = {}
    pil_images = []
    for t in inference_times:
        inference_time = t.item()
        latent_dict[inference_time] = latent
        with torch.no_grad():
            np_img = latent2image(latent, vae, return_type='np')
        pil_img = Image.fromarray(np_img)
        pil_images.append(pil_img)
        pil_img.save(os.path.join(base_folder_dir, f'with_con_with_self_qkv_recon_{t.item()}.png'))
        # ----------------------------------------------------------------------------
        time_steps.append(inference_time)
        noise_pred = call_unet(unet, latent, t, cond_embeddings, t.item(), None)
        #noise_pred = call_unet(unet, latent, t, uncond_embeddings, t.item(), None)
        latent = prev_step(noise_pred, t.item(), latent, scheduler)
        all_latent.append(latent)
    return all_latent, time_steps, pil_images
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

def main(args) :

    print(f' \n step 1. make stable diffusion model')
    if args.process_title:
        setproctitle(args.process_title)
    else:
        setproctitle('parksooyeon')

    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)

    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f" (1.0.1) logging")
    if args.log_with == 'wandb' :
        wandb.init(project=args.wandb_init_name, name=args.wandb_run_name)

    print(f" (1.0.2) save directory and save config")
    save_base_dir = args.output_dir
    _, folder_name = os.path.split(save_base_dir)
    record_save_dir = os.path.join(args.output_dir, "record")
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    base_folder_dir = os.path.join(args.folder_name)
    os.makedirs(base_folder_dir, exist_ok=True)

    print(f" (1.0.3) save directory and save config")
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    print(f' (1.1) tokenizer')
    tokenizer = train_util.load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

    print(f' (1.2) SD')
    text_encoder, vae, unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype,
                                                                                          args.device,
                                                                                          unet_use_linear_projection_in_v2=False, )
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    print(f' (1.3) network')
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
                                                vae, text_encoder, unet, neuron_dropout=args.network_dropout, **net_kwargs, )
    print(f' (1.3.4) apply trained state dict')
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)

    print(f' (1.3.5) register attention storer')


    print(f' (1.4) scheduler')
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

    # scheduler:
    SCHEDULER_LINEAR_START = 0.00085
    SCHEDULER_LINEAR_END = 0.0120
    SCHEDULER_TIMESTEPS = 1000
    SCHEDLER_SCHEDULE = "scaled_linear"
    scheduler = scheduler_cls(num_train_timesteps=SCHEDULER_TIMESTEPS,
                              beta_start=SCHEDULER_LINEAR_START,
                              beta_end=SCHEDULER_LINEAR_END,
                              beta_schedule=SCHEDLER_SCHEDULE,)
    scheduler.set_timesteps(args.num_ddim_steps)
    inference_times = scheduler.timesteps

    print(f' (1.4) model to accelerator device')
    device = args.device
    if len(text_encoders) > 1:
        unet, t_enc1, t_enc2 = unet.to(device), text_encoders[0].to(device), text_encoders[1].to(device)
        text_encoder = text_encoders = [t_enc1, t_enc2]
        del t_enc1, t_enc2
    else:
        unet, text_encoder = unet.to(device), text_encoder.to(device)
        text_encoders = [text_encoder]
    network.to(device)

    print(f' \n step 2. ground-truth image preparing')
    print(f' (2.1) prompt condition')
    prompt = args.prompt
    context = init_prompt(tokenizer, text_encoder, device, prompt)
    print(f' (2.2) image condition')
    concept_img_dirs = os.listdir(args.concept_image_folder)
    print(f' (2.3) inverting as saving self k&v')
    self_q, self_k, self_v, cross_q, cross_k, cross_v = {}, {}, {}, {}, {}, {}
    for concept_img in concept_img_dirs :
        print(f' (2.3.1) inversion')
        attention_storer = AttentionStore()
        register_attention_control(unet, attention_storer)
        concept_img_dir = os.path.join(args.concept_image_folder, concept_img)
        image_gt_np = load_512(concept_img_dir)
        latent = image2latent(image_gt_np, vae, device, weight_dtype)
        ddim_latents, time_steps, pil_images = ddim_loop(latent, context, inference_times, scheduler, unet, vae,base_folder_dir)
        layer_names = attention_storer.self_query_store.keys()
        self_query_collection = attention_storer.self_query_store
        self_key_collection = attention_storer.self_key_store
        self_value_collection = attention_storer.self_value_store
        self_query_dict, self_key_dict, self_value_dict = {}, {}, {}
        #cross_query_dict, cross_key_dict, cross_value_dict = {}, {}, {}
        for layer in layer_names:
            self_query_list = attention_storer.self_query_store[layer]
            self_key_list = attention_storer.self_key_store[layer]
            self_value_list = attention_storer.self_value_store[layer]
            #cross_layer = layer.replace('attn1', 'attn2')
            #cross_query_list = attention_storer.cross_query_store[cross_layer]
            #cross_key_list = attention_storer.cross_key_store[cross_layer]
            #cross_value_list = attention_storer.cross_value_store[cross_layer]
            i = 0
            #for self_query, self_key, self_value, cross_query, cross_key, cross_value in zip(self_query_list, self_key_list, self_value_list,
            #                                                                                 cross_query_list,cross_key_list,cross_value_list,) :
            for self_query, self_key, self_value in zip(self_query_list, self_key_list, self_value_list) :
                time_step = time_steps[i]
                if time_step not in self_query_dict.keys() :
                    self_query_dict[time_step] = {}
                    self_query_dict[time_step][layer] = self_query
                else :
                    self_query_dict[time_step][layer] = self_query

                if time_step not in self_key_dict.keys() :
                    self_key_dict[time_step] = {}
                    self_key_dict[time_step][layer] = self_key
                else :
                    self_key_dict[time_step][layer] = self_key

                if time_step not in self_value_dict.keys() :
                    self_value_dict[time_step] = {}
                    self_value_dict[time_step][layer] = self_value
                else :
                    self_value_dict[time_step][layer] = self_value
                """
                if time_step not in cross_query_dict.keys() :
                    cross_query_dict[time_step] = {}
                    cross_query_dict[time_step][layer] = cross_query
                else :
                    cross_query_dict[time_step][layer] = cross_query
                if time_step not in cross_key_dict.keys() :
                    cross_key_dict[time_step] = {}
                    cross_key_dict[time_step][layer] = cross_key
                else :
                    cross_key_dict[time_step][layer] = cross_key
                if time_step not in cross_value_dict.keys() :
                    cross_value_dict[time_step] = {}
                    cross_value_dict[time_step][layer] = cross_value
                else :
                    cross_value_dict[time_step][layer] = cross_value
                """
                i += 1
        concept_img_name = os.path.splitext(concept_img)[0]
        self_q[concept_img_name] = self_query_dict
        self_k[concept_img_name] = self_key_dict
        self_v[concept_img_name] = self_value_dict
        #cross_q[concept_img_name] = cross_query_dict
        #cross_k[concept_img_name] = cross_key_dict
        #cross_v[concept_img_name] = cross_value_dict
        attention_storer.reset()
        print(f' (2.3.2) reconstruction with correcting')
        unregister_attention_control(unet, attention_storer)
        start_latent = ddim_latents[-1]
        ddim_latents, time_steps, pil_images = recon_loop(start_latent,
                                                          context,
                                                          inference_times,
                                                          scheduler, unet, vae,
                                                          self_query_dict,
                                                          self_key_dict,
                                                          self_value_dict,
                                                          base_folder_dir)
        break
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う", )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--process_title", type=str, default='parksooyeon')
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
    parser.add_argument("--prompt", type=str,
                        default = 'teddy bear, wearing like a super hero')
    parser.add_argument("--negative_prompt", type=str,
                        default = 'low quality, worst quality, bad anatomy,bad composition, poor, low effort')
    parser.add_argument("--concept_image_folder", type=str)
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--folder_name", type=str)
    
    
    

    parser.add_argument("--max_self_input_time", type=int, default=10)
    parser.add_argument("--min_value", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--self_key_control", action='store_true')
    
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    main(args)
