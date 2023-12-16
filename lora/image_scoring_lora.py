import argparse
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

def register_attention_control(unet : nn.Module, controller:AttentionStore):

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
                                                         device=query.device), query,key.transpose(-1, -2),beta=0,alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)
            if is_cross_attention:
                if trg_indexs_list is not None :
                    org_attention_probs, rec_attention_probs = attention_probs.chunk(2, dim=0)
                    batch_num = len(trg_indexs_list)
                    org_attention_probs_batch = torch.chunk(org_attention_probs, batch_num, dim=0)
                    rec_attention_probs_batch = torch.chunk(rec_attention_probs, batch_num, dim=0)
                    for batch_idx, (org_prob,rec_prob) in enumerate(zip(org_attention_probs_batch,rec_attention_probs_batch)) :
                        batch_trg_index = trg_indexs_list[batch_idx] # two times
                        for word_idx in batch_trg_index :
                            word_idx = int(word_idx)
                            org_attn_vector = org_prob[:, :, word_idx] # bad
                            rec_attn_vector = rec_prob[:, :, word_idx]          # good
                            attention_diff = abs(rec_attn_vector-org_attn_vector)#.mean()
                            controller.save(attention_diff, layer_name)
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


def generate_text_embedding(caption, class_caption, tokenizer):
    cls_token = 49406
    pad_token = 49407
    token_input = tokenizer([class_caption], padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt", )  # token_input = 24215
    token_ids = token_input.input_ids[0]
    token_attns = token_input.attention_mask[0]
    trg_token_id = []
    for token_id, token_attn in zip(token_ids, token_attns):
        if token_id != cls_token and token_id != pad_token and token_attn == 1:
            # token_id = 24215
            trg_token_id.append(token_id)
    text_input = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt", )
    token_ids = text_input.input_ids
    attns = text_input.attention_mask
    for token_id, attn in zip(token_ids, attns):
        trg_indexs = []
        for i, id in enumerate(token_id):
            if id in trg_token_id:
                trg_indexs.append(i)
    return trg_indexs


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


def prev_step(model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray],
              scheduler):
    timestep, prev_timestep = timestep, max(
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 0)
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_matrix = torch.ones_like(model_output) * alpha_prod_t

    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
    alpha_prod_t_prev_matrix = torch.ones_like(model_output) * alpha_prod_t_prev
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_matrix = torch.ones_like(model_output) * beta_prod_t

    # prev_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    prev_original_sample = (sample - beta_prod_t_matrix ** 0.5 * model_output) / alpha_prod_t_matrix ** 0.5
    # prev_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
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
    flip_times = inference_times
    repeat_time = 0
    for i, t in enumerate(flip_times[:-1]):
        if repeat_time < args.repeat_time:
            next_time = flip_times[i + 1].item()
            latent_dict[int(t.item())] = latent
            time_steps.append(t.item())
            noise_pred = call_unet(unet, latent, t, uncond_embeddings, None, None)
            noise_pred_dict[int(t.item())] = noise_pred
            latent = next_step(noise_pred, int(t.item()), latent, scheduler)

            np_img = latent2image(latent, vae, return_type='np')
            pil_img = Image.fromarray(np_img)
            pil_images.append(pil_img)
            # pil_img.save(os.path.join(base_folder_dir, f'noising_{next_time}.png'))
            repeat_time += 1
    # time_steps.append(next_time)
    latent_dict[int(next_time)] = latent
    latent_dict_keys = latent_dict.keys()
    return latent_dict, time_steps, pil_images


@torch.no_grad()
def recon_loop(latent_dict, context, inference_times, scheduler, unet, vae, base_folder_dir, vae_factor_dict):
    uncon, con = context.chunk(2)
    if inference_times[0] < inference_times[1]:
        inference_times.reverse()
    inference_times = inference_times[1:]
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
    latent_y = latent
    for i, t in enumerate(inference_times[:-1]):
        prev_time = int(inference_times[i + 1])
        time_steps.append(int(t))

        with torch.no_grad():
            noise_pred = call_unet(unet, latent, t, con, t, prev_time)
            latent = prev_step(noise_pred, int(t), latent, scheduler)
            factor = float(vae_factor_dict[prev_time])
            if args.using_customizing_scheduling:
                np_img = latent2image_customizing(latent, vae, factor, return_type='np')
            else:
                np_img = latent2image(latent, vae, return_type='np')
        if prev_time == 0:
            pil_img = Image.fromarray(np_img)
            pil_images.append(pil_img)
            pil_img.save(os.path.join(base_folder_dir, f'recon_{prev_time}.png'))
        all_latent_dict[prev_time] = latent
    time_steps.append(prev_time)
    return all_latent_dict, time_steps, pil_images


def main(args):
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

    print(f' \n step 2. make stable diffusion model')
    device = args.device
    print(f' (2.1) tokenizer')
    tokenizer = train_util.load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
    print(f' (2.2) SD')
    text_encoder, vae, unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype, device,
                                                                                          unet_use_linear_projection_in_v2=False, )
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    print(f' (2.3) scheduler')
    sched_init_args = {}
    scheduler_cls = DDIMScheduler
    if args.v_parameterization: sched_init_args["prediction_type"] = "v_prediction "
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
    if len(text_encoders) > 1:
        unet, t_enc1, t_enc2 = unet.to(device), text_encoders[0].to(device), text_encoders[1].to(device)
        text_encoder = [t_enc1, t_enc2]
        del t_enc1, t_enc2
    else:
        unet, text_encoder = unet.to(device), text_encoder.to(device)
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
    print(f' (2.3.4) apply trained state dict')
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
    network.to(device)

    print(f' (2.4) scheduling factors')
    vae_factor_dict = r'../result/inference_decoding_factor_txt.txt'

    with open(vae_factor_dict, 'r') as f:
        content = f.readlines()
    inference_decoding_factor = {}
    for line in content:
        line = line.strip()
        line = line.split(' : ')
        t, f = int(line[0]), float(line[1])
        inference_decoding_factor[t] = f

    print(f' \n step 3. Normal Sample')
    base_dir = '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/recon_check/train/noising_inverse_unet_denoising_lora_50_model_epoch_3/044/final_time_980'
    org_img_dir = os.path.join(base_dir, 'original_sample.png')
    rec_img_dir = os.path.join(base_dir, 'recon_0.png')
    prompt = 'good'
    context = init_prompt(tokenizer, text_encoder, device, prompt)
    uncon, con = context.chunk(2)
    vae_scale_factor = 0.18215

    attention_storer = AttentionStore()
    register_attention_control(unet, attention_storer)

    with torch.no_grad():

        trg_indexs_list = generate_text_embedding(prompt, prompt, tokenizer)
        trg_indexs_list = [trg_indexs_list]
        org_latent = image2latent(load_512(org_img_dir), vae,device, weight_dtype)
        org_latent = org_latent * vae_scale_factor
        rec_latent = image2latent(load_512(rec_img_dir),vae,device, weight_dtype)
        rec_latent = rec_latent * vae_scale_factor

        input_latent = torch.cat([org_latent, rec_latent], dim=0)
        input_cond = torch.cat([con, con], dim=0)

        noise_pred = unet(input_latent,
                          0,
                          input_cond,
                          trg_indexs_list=trg_indexs_list,
                          mask_imgs=None).sample
        heatmap_stores = attention_storer.heatmap_store
        attention_storer.reset()
        layer_names = heatmap_stores.keys()
        for layer_name in layer_names:
            heatmap_vector = heatmap_stores[layer_name][0]
            max_score = max(heatmap_vector)
            print(f'{layer_name} : max_score : {max_score}')

    print(f' \n step 4. Test Sample')
    base_dir = '/data7/sooyeon/Lora/OODLora/result/MVTec_experiment/bagel/recon_check/test/crack/noising_inverse_unet_denoising_lora_inference_time_50_model_epoch_3/010/final_time_980'
    org_img_dir = os.path.join(base_dir, 'original_sample.png')
    rec_img_dir = os.path.join(base_dir, 'recon_0.png')

    context = init_prompt(tokenizer, text_encoder, device, prompt)
    uncon, con = context.chunk(2)
    vae_scale_factor = 0.18215
    attention_storer = AttentionStore()
    register_attention_control(unet, attention_storer)

    with torch.no_grad():
        trg_indexs_list = generate_text_embedding(prompt, prompt, tokenizer)
        trg_indexs_list = [trg_indexs_list]
        org_latent = image2latent(load_512(org_img_dir), vae, device, weight_dtype)
        org_latent = org_latent * vae_scale_factor
        rec_latent = image2latent(load_512(rec_img_dir), vae, device, weight_dtype)
        rec_latent = rec_latent * vae_scale_factor
        input_latent = torch.cat([org_latent, rec_latent], dim=0)
        input_cond = torch.cat([con, con], dim=0)
        noise_pred = unet(input_latent,
                          0,
                          input_cond,
                          trg_indexs_list=trg_indexs_list,
                          mask_imgs=None).sample
        heatmap_stores = attention_storer.heatmap_store
        attention_storer.reset()
        layer_names = heatmap_stores.keys()
        for layer_name in layer_names:
            heatmap_vector = heatmap_stores[layer_name][0]
            max_score = max(heatmap_vector)
            print(f'{layer_name} : max_score : {max_score}')




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
    parser.add_argument("--self_key_control", action='store_true')
    parser.add_argument("--inversion_experiment", action="store_true", )
    parser.add_argument("--repeat_time", type=int, default=1)
    parser.add_argument("--self_attn_threshold_time", type=int, default=1)
    parser.add_argument("--using_customizing_scheduling", action="store_true", )
    parser.add_argument("--final_time", type=int, default=600)
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    main(args)