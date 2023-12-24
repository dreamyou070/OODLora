import argparse
from accelerate.utils import set_seed
import os
import random
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
import torch
from PIL import Image
import sys, importlib
from typing import Union
import numpy as np
from utils.image_utils import image2latent, customizing_image2latent, load_image, latent2image
import shutil
from attention_store import AttentionStore
from torch import nn
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
                       LMSDiscreteScheduler,PNDMScheduler,EulerDiscreteScheduler,HeunDiscreteScheduler, KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler)
from diffusers import DDIMScheduler

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


def call_unet(unet, noisy_latents, timesteps,
              text_conds, trg_indexs_list, mask_imgs):
    noise_pred = unet(noisy_latents,
                      timesteps,
                      text_conds,
                      trg_indexs_list=trg_indexs_list,
                      mask_imgs=mask_imgs,).sample
    return noise_pred


@torch.no_grad()
def ddim_loop(latent, context, inference_times, scheduler, unet, vae, base_folder_dir, is_org):
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
    repeat_time = 0
    for i, t in enumerate(flip_times[:-1]):
        if repeat_time < args.repeat_time:
            next_time = flip_times[i + 1]
            latent_dict[int(t)] = latent
            time_steps.append(t)
            noise_pred = call_unet(unet, latent, t, cond_embeddings, None, None)
            noise_pred_dict[int(t)] = noise_pred
            latent = next_step(noise_pred, int(t), latent, scheduler)
            np_img = latent2image(latent, vae, return_type='np')
            pil_img = Image.fromarray(np_img)
            pil_images.append(pil_img)
            if is_org:
                pil_img.save(os.path.join(base_folder_dir, f'noising_{next_time}.png'))
            else :
                pil_img.save(os.path.join(base_folder_dir, f'student_noising_{next_time}.png'))
            repeat_time += 1
    time_steps.append(next_time)
    latent_dict[int(next_time)] = latent
    return latent_dict, time_steps, pil_images


@torch.no_grad()
def recon_loop(latent_dict, start_latent, context, inference_times, scheduler, unet, vae, base_folder_dir, controller):
    if context.shape[0] == 2:
        uncon, con = context.chunk(2)
    else:
        con = context
    # inference_times = [100,80, ... 0]
    latent = start_latent
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
        z_latent = latent_dict[inference_times[i]]
        x_latent = latent
        prev_time = int(inference_times[i + 1])
        time_steps.append(int(t))
        with torch.no_grad():
            input_latent = torch.cat([z_latent, x_latent], dim=0)
            input_cond = torch.cat([con, con], dim=0)
            trg_indexs_list = [[1]]
            noise_pred = call_unet(unet,
                                   input_latent,
                                   t,
                                   input_cond,
                                   trg_indexs_list, None)



            mask_dict = controller.step_store
            controller.reset()
            layer_names = mask_dict.keys()
            mask_list = []
            import torchvision
            totensor = torchvision.transforms.ToTensor()
            for layer_name in layer_names:
                mask_torch = mask_dict[layer_name][0] # head, pix_num, 1
                if mask_torch.dim() == 2 :
                    mask_torch = mask_torch.unsqueeze(-1)
                head, pix_num, _ = mask_torch.shape
                res = int(pix_num ** 0.5)
                cross_maps = mask_torch.reshape(head, res, res, mask_torch.shape[-1])
                cross_maps = cross_maps.mean([-1])
                cross_maps = cross_maps.mean([0])
                image = cross_maps.numpy().astype(np.uint8)
                mask_list.append(totensor(Image.fromarray(image).resize((64, 64))))
            mask = torch.stack(mask_list, dim=0).mean([0]).unsqueeze(0)
            print(f'mask : {mask.shape}')
            y_latent = z_latent * (1-mask) + x_latent * (mask) # 1,4,64,64
            y_noise_pred = call_unet(unet, y_latent, t, con, None, None)
            controller.reset()
            # --------------------- mask --------------------- #
            latent = prev_step(y_noise_pred, t, y_latent, scheduler)
            np_img = latent2image(latent, vae, return_type='np')
        #if prev_time == 0:
            pil_img = Image.fromarray(np_img)
            pil_images.append(pil_img)
            pil_img.save(os.path.join(base_folder_dir, f'recon_{prev_time}.png'))
        all_latent_dict[prev_time] = latent
    time_steps.append(prev_time)
    return all_latent_dict, time_steps, pil_images


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
                        attention_diff = (masked_attn_vector - org_attn_vector) # head, pix_num, 1
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
    output_dir = args.output_dir
    if args.use_binary_mask :
        parent, child = os.path.split(args.output_dir)
        output_dir = os.path.join(parent, f'{child}_binary_mask_thred_{args.mask_thredhold}')
    parent, network_dir = os.path.split(args.network_weights)
    model_name = os.path.splitext(network_dir)[0]
    if 'last' not in model_name:
        model_epoch = int(model_name.split('-')[-1])
    else:
        model_epoch = 'last'



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
    vae.to(accelerator.device, dtype=vae_dtype)

    print(f' (2.3) vae student model')
    from diffusers import AutoencoderKL
    from STTraining import Encoder_Student
    from utils.model_util import get_state_dict
    student_vae = AutoencoderKL.from_config(vae.config)
    student = Encoder_Student(student_vae.encoder, student_vae.quant_conv)
    student.load_state_dict(get_state_dict(args.student_pretrained_dir), strict=True)
    student.requires_grad_(False)
    student.eval()
    student.to(accelerator.device, dtype=vae_dtype)
    student_epoch = os.path.split(args.student_pretrained_dir)[-1]
    student_epoch = os.path.splitext(student_epoch)[0]
    student_epoch = int(student_epoch.split('_')[-1])
    print(f'student_epoch: {student_epoch}')
    output_dir = os.path.join(output_dir, f'lora-epoch-{model_epoch}-student-epoch-{student_epoch}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'final output dir : {output_dir}')

    print(f' (2.4) scheduler')
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
        network = network_module.create_network(1.0,args.network_dim,
                                                args.network_alpha,
                                                vae, text_encoder, unet, neuron_dropout=args.network_dropout,
                                                **net_kwargs, )
    print(f' (2.5.3) apply trained state dict')
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
    network.to(device)
    controller = AttentionStore()
    register_attention_control(unet, controller)

    print(f' \n step 3. ground-truth image preparing')
    print(f' (3.1) prompt condition')
    prompt = args.prompt

    context = init_prompt(tokenizer, text_encoder, device, prompt)

    print(f' (3.2) test images')
    trg_h, trg_w = args.resolution
    train_img_folder = os.path.join(args.concept_image_folder, 'train/bad')
    train_mask_folder = os.path.join(args.concept_image_folder, 'train/gt')
    classes = os.listdir(train_img_folder)

    thredhold = args.mask_thredhold

    for class_name in classes:
        repeat, c_name = class_name.split('_')

        class_base_folder = os.path.join(output_dir, c_name)
        os.makedirs(class_base_folder, exist_ok=True)

        image_folder = os.path.join(train_img_folder, class_name)
        if '_' in class_name:
            class_name =  '_'.join(class_name.split('_')[1:])

        invers_context = init_prompt(tokenizer, invers_text_encoder, device, f'a photo of {c_name}')
        inv_unc, inv_c = invers_context.chunk(2)
        mask_folder = os.path.join(train_mask_folder, class_name)

        train_images = os.listdir(image_folder)
        for j, train_img in enumerate(train_images):
            if j < 1 :
                name, ext = os.path.splitext(train_img)

                train_img_dir = os.path.join(image_folder, train_img)

                shutil.copy(train_img_dir, os.path.join(class_base_folder, train_img))
                if 'good' not in class_name:
                    mask_img_dir = os.path.join(mask_folder, train_img)
                    shutil.copy(mask_img_dir, os.path.join(class_base_folder, f'{name}_mask{ext}'))

                print(f' (2.3.1) inversion')
                image_gt_np = load_image(train_img_dir, trg_h = int(trg_h), trg_w =int(trg_w))
                with torch.no_grad():
                    org_vae_latent = image2latent(image_gt_np, vae, device=device, weight_dtype=weight_dtype)
                    st_latent = customizing_image2latent(image_gt_np, student, device=device, weight_dtype=weight_dtype)
                    inf_time = inference_times.tolist()
                    inf_time.reverse() # [0,20,40,60,80,100 , ... 980]
                    org_latent_dict, time_steps, pil_images = ddim_loop(latent=org_vae_latent,
                                                                        context=inv_c,
                                                                        inference_times=inf_time,
                                                                        scheduler=scheduler,
                                                                        unet=invers_unet,
                                                                        vae=vae,
                                                                        base_folder_dir=class_base_folder,
                                                                        is_org = True)
                    latent_dict, time_steps, pil_images = ddim_loop(latent=st_latent,
                                                                        context=inv_c,
                                                                        inference_times=inf_time,
                                                                        scheduler=scheduler,
                                                                        unet=invers_unet,
                                                                        vae=vae,
                                                                        base_folder_dir=class_base_folder,
                                                                        is_org = False)

                    base_num = 40
                    noising_time = inference_times[base_num] # 100
                    recon_times = inference_times[base_num:].tolist()
                    st_noise_latent = latent_dict[int(noising_time.item())]
                    recon_loop(org_latent_dict,
                               start_latent = st_noise_latent,
                               context = context,
                               inference_times = recon_times,
                               scheduler = scheduler,
                               unet = unet,
                               vae = vae,
                               base_folder_dir = class_base_folder,
                               controller = controller,)






                    """
                    
                    print(f'inference_times : {inference_times}')
                    
                    org_noise_latent = scheduler.add_noise(original_samples = org_vae_latent, noise = standard_noise, timesteps = torch.tensor(int(noising_time)))
                    
                    Image.fromarray(latent2image(org_noise_latent, vae, return_type='np')).save(os.path.join(class_base_folder,
                                                                                                      f'{name}_org_vae_noise_latent_{noising_time}{ext}'))
                    Image.fromarray(latent2image(st_noise_latent, vae, return_type='np')).save(os.path.join(class_base_folder,
                                                                                                            f'{name}_st_vae_noise_latent_{noising_time}{ext}'))
                    inf_times = inference_times[base_num:].tolist() # from 780
                    inf_times.reverse()
                    org_recon_loop(org_noise_latent, # 780 noise latent
                                   inv_c,
                                   inf_times,
                                   scheduler,
                                   invers_unet, vae, class_base_folder)
                    """


    """
    mse = ((st_latent - org_vae_latent).square() * 2) - thredhold
    mse_threshold = mse < 0  # if true = 1, false = 0 # if true -> bad
    mse_threshold = (mse_threshold.float())  # 0 = background, 1 = bad point


    new_latent = org_vae_latent * (1-mse_threshold) + st_latent * mse_threshold
    mask_np_img = latent2image(new_latent, vae, return_type='np')
    pil_img = Image.fromarray(mask_np_img)
    pil_img.save(os.path.join(save_base_dir, f'vae_masked_{test_img}'))
    
    
    inference_times = torch.cat([torch.tensor([999]), scheduler.timesteps, ], dim=0)
    flip_times = torch.flip(inference_times, dims=[0])  # [0,20, ..., 980]
    #original_latent = latent.clone().detach()
    original_latent = org_vae_latent.clone().detach()
    for ii, final_time in enumerate(flip_times[1:]):

        if final_time.item() == args.final_time:
            timewise_save_base_folder = os.path.join(save_base_dir, f'final_time_{final_time.item()}')
            print(f' - save_base_folder : {timewise_save_base_folder}')
            os.makedirs(timewise_save_base_folder, exist_ok=True)

            mask_pil = Image.open(mask_img_dir).resize((512, 512)).convert('RGB')
            mask_pil.save(os.path.join(timewise_save_base_folder, 'mask.png'))

            org_pil = Image.open(train_img_dir).resize((512, 512)).convert('RGB')
            org_pil.save(os.path.join(timewise_save_base_folder, 'org.png'))

            np_img = latent2image(st_latent, vae, return_type='np')
            pil_img = Image.fromarray(np_img)
            pil_img.save(os.path.join(timewise_save_base_folder, f'vae_recon.png'))

            if args.use_binary_mask :
                latent = original_latent
            else :
                latent = st_latent

            latent_dict, time_steps, pil_images = ddim_loop(latent=latent,
                                                            context=invers_context,
                                                            inference_times=flip_times[:ii + 2],
                                                            scheduler=scheduler,
                                                            unet=invers_unet,
                                                            vae=vae,
                                                            base_folder_dir=timewise_save_base_folder,)
            if args.use_binary_mask :
                torch.manual_seed(args.seed)
                start_latent = scheduler.add_noise(original_samples = st_latent,
                                                   noise = torch.randn(original_latent.shape, dtype=weight_dtype).to(st_latent.device),
                                                   timesteps = torch.tensor(time_steps[-1], dtype=torch.int8).to(st_latent.device),
                                                   )


            else :
                start_latent = latent_dict[int(time_steps[-1])]
            time_steps.reverse()
            print(f'time_steps : {time_steps}')
            context = init_prompt(tokenizer, text_encoder, device, prompt)

            print(f' (2.3.2) recon')
            if args.use_binary_mask :
                mask = mse_threshold
            else :
                mask = None
            recon_latent_dict, _, _ = recon_loop(latent_dict=latent_dict,
                                                     start_latent = start_latent,
                                                     context=context,
                                                     inference_times=time_steps,  # [20,0]
                                                     scheduler=scheduler,
                                                     unet=unet,
                                                     vae=vae,
                                                     base_folder_dir=timewise_save_base_folder,
                                                     mask=mask)
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
    parser.add_argument("--repeat_time", type=int, default=1)
    parser.add_argument("--final_time", type=int, default = 600)
    parser.add_argument("--student_pretrained_dir", type=str)
    parser.add_argument("--mask_thredhold", type=float, default = 0.5)
    parser.add_argument("--use_binary_mask", action = 'store_true')
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    main(args)