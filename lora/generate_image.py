import argparse
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
import os
import random
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
import torch
from PIL import Image
import sys, importlib
from utils.model_utils import call_unet
import numpy as np
from utils.image_utils import image2latent, customizing_image2latent, load_image
from utils.scheduling_utils import get_scheduler, ddim_loop, recon_loop
from utils.model_utils import get_state_dict, init_prompt
import shutil
from attention_store import AttentionStore
import torch.nn as nn
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None

def register_attention_control(unet: nn.Module, controller: AttentionStore,  mask_threshold: float = 1):  # if mask_threshold is 1, use itself

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
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                            device=query.device), query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)
            if is_cross_attention and trg_indexs_list is not None:
                masked_attention_probs, org_attention_probs = attention_probs.chunk(2, dim=0)
                batch_num = len(trg_indexs_list)
                attention_probs_batch = torch.chunk(org_attention_probs, batch_num, dim=0)
                masked_attention_probs_batch = torch.chunk(masked_attention_probs, batch_num, dim=0)
                vector_diff_list = []
                for batch_idx, (attention_prob, masked_attention_prob) in enumerate(zip(attention_probs_batch, masked_attention_probs_batch)):
                    batch_trg_index = trg_indexs_list[batch_idx]  # two times
                    for word_idx in batch_trg_index:
                        word_idx = int(word_idx)
                        masked_attn_vector = masked_attention_prob[:, :, word_idx] # head, pix_num, 1
                        org_attn_vector = attention_prob[:, :, word_idx]
                        attention_diff = torch.nn.functional.mse_loss(masked_attn_vector, org_attn_vector,
                                                           reduction='none')
                        controller.store(attention_diff,layer_name)
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

    print(f" (1.2) save directory and save config")
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    print(f" (1.3) save dir")
    weights = os.listdir(args.network_weights)
    parent, _ = os.path.split(args.network_weights)
    output_dir = os.path.join(parent, 'image_generating')
    os.makedirs(output_dir, exist_ok=True)

    for weight in weights:
        weight_dir = os.path.join(args.network_weights, weight)
        model_name = os.path.splitext(weight)[0]
        if 'last' not in model_name:
            model_epoch = int(model_name.split('-')[-1])
        else:
            model_epoch = 'last'


        print(f' \n step 2. make stable diffusion model')
        device = accelerator.device
        print(f' (2.1) tokenizer')
        tokenizer = train_util.load_tokenizer(args)
        print(f' (2.2) SD')
        invers_text_encoder, vae, invers_unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype,device,
                                                                                                            unet_use_linear_projection_in_v2=False, )
        invers_text_encoders = invers_text_encoder if isinstance(invers_text_encoder, list) else [invers_text_encoder]
        text_encoder, vae, unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype, device,
                                                                                              unet_use_linear_projection_in_v2=False, )
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        vae.to(accelerator.device, dtype=vae_dtype)


        print(f' (2.4) scheduler')
        scheduler_cls = get_scheduler(args.sample_sampler, args.v_parameterization)[0]
        scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps, beta_start=args.scheduler_linear_start,
                                  beta_end=args.scheduler_linear_end, beta_schedule=args.scheduler_schedule)
        scheduler.set_timesteps(args.num_ddim_steps)
        inference_times = scheduler.timesteps

        print(f' (2.4.+) model to accelerator device')
        if len(invers_text_encoders) > 1:
            invers_unet, invers_t_enc1, invers_t_enc2 = invers_unet.to(device), invers_text_encoders[0].to(device),invers_text_encoders[1].to(device)
            invers_text_encoder = [invers_t_enc1, invers_t_enc2]
            del invers_t_enc1, invers_t_enc2
            unet, t_enc1, t_enc2 = unet.to(device), text_encoders[0].to(device), text_encoders[1].to(device)
            text_encoder = [t_enc1, t_enc2]
            del t_enc1, t_enc2
        else:
            invers_unet, invers_text_encoder = invers_unet.to(device), invers_text_encoder.to(device)
            unet, text_encoder = unet.to(device), text_encoder.to(device)

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
            network, _ = network_module.create_network_from_weights(1, weight_dir, vae, text_encoder, unet,
                                                                    **net_kwargs)
        else:
            network = network_module.create_network(1.0,args.network_dim, args.network_alpha, vae, text_encoder, unet,
                                                    neuron_dropout=args.network_dropout, **net_kwargs, )
        print(f' (2.5.3) apply trained state dict')
        network.apply_to(text_encoder, unet, True, True)
        info = network.load_weights(weight_dir)
        network.to(device)
        controller = AttentionStore()
        register_attention_control(unet, controller)

        print(f' \n step 3. ground-truth image preparing')
        print(f' (3.1) prompt condition')
        prompts = ['hole','crack','contamination']
        for p in prompts:
            save_dir = os.path.join(output_dir, f'lora_epoch_{model_epoch}_prompt_{p}_guidance_scale_{args.guidance_scale}')
            os.makedirs(save_dir, exist_ok=True)
            context = init_prompt(tokenizer, text_encoder, device, p)
            print(f' (3.2) train images')
            from utils.scheduling_utils import prev_step
            from utils.image_utils import latent2image
            # inference_times = [100,80, ... 0]
            latent = torch.randn(1,4,64,64)
            for i, t in enumerate(inference_times[:-1]):
                prev_time = int(inference_times[i + 1])
                with torch.no_grad():
                    input_latent = torch.cat([latent,latent], dim=0).to(accelerator.device, weight_dtype)
                    input_cond = context
                    noise_pred = call_unet(unet, input_latent, t, input_cond, None, None)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latent = prev_step(noise_pred, t, latent.to(accelerator.device, weight_dtype), scheduler)
                    pil_img = Image.fromarray(latent2image(latent, vae, return_type='np'))
                    #pil_img.save(os.path.join(save_dir, f'gen_{t}.png'))
            pil_img = Image.fromarray(latent2image(latent, vae, return_type='np'))
            pil_img.save(os.path.join(save_dir, f'gen_{prev_time}.png'))


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
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision", )
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train")
    parser.add_argument("--base_weights", type=str, default=None, nargs="*",
                        help="network weights to merge into the model before training", )
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*",
                        help="multiplier for network weights to merge into the model before training", )
    parser.add_argument("--network_dim", type=int, default=None,
                        help="network dimensions (depends on each network)")
    parser.add_argument("--network_alpha", type=float, default=1,
                        help="alpha for LoRA weight scaling, default 1", )
    parser.add_argument("--network_dropout", type=float, default=None,)
    parser.add_argument("--network_args", type=str, default=None, nargs="*",)
    parser.add_argument("--dim_from_weights", action="store_true",)
    parser.add_argument("--network_weights", type=str, default=None,help="pretrained weights for network")
    parser.add_argument("--concept_image", type=str,
                        default = '/data7/sooyeon/MyData/perfusion_dataset/td_100/100_td/td_1.jpg')
    parser.add_argument("--prompt", type=str, default = 'teddy bear, wearing like a super hero')
    parser.add_argument("--concept_image_folder", type=str)
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--scheduler_linear_start", type=float, default=0.00085)
    parser.add_argument("--scheduler_linear_end", type=float, default=0.012)
    parser.add_argument("--scheduler_timesteps", type=int, default=1000)
    parser.add_argument("--scheduler_schedule", type=str, default="scaled_linear")
    parser.add_argument("--unet_only_inference_times", type=int, default = 30)
    parser.add_argument("--student_pretrained_dir", type=str)
    parser.add_argument("--mask_thredhold", type=float, default = 0.5)
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    args = parser.parse_args()
    main(args)