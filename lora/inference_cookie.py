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
import numpy as np
from utils.image_utils import image2latent, customizing_image2latent, load_image
from utils.scheduling_utils import get_scheduler, ddim_loop, recon_loop
from utils.model_utils import get_state_dict, init_prompt
import shutil
from attention_store import AttentionStore
import torch.nn as nn
from utils.model_utils import call_unet
import csv
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None


def register_attention_control(unet: nn.Module, controller: AttentionStore,
                               mask_threshold: float = 1):  # if mask_threshold is 1, use itself

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
            if is_cross_attention:

                controller.store(attention_probs[:,:,:3], layer_name)

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

    parent = os.path.split(args.network_weights)[0] # unique_folder,
    args.output_dir = os.path.join(parent, f'all_res_check_cross_attention_map')
    os.makedirs(args.output_dir, exist_ok=True)
    record_output_dir = os.path.join(parent, 'all_res_check_score_record')
    print(f'base record dir : {record_output_dir}')
    os.makedirs(record_output_dir, exist_ok=True)


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

    # ------------------------------------------------ test check -------------------------------------------------------- #
    print(f" (1.3) save dir")

    test_set_output_dir = os.path.join(args.output_dir, 'test_set')
    os.makedirs(test_set_output_dir, exist_ok=True)
    test_set_record_dir = os.path.join(record_output_dir, 'test_set')
    os.makedirs(test_set_record_dir, exist_ok=True)

    network_weights = os.listdir(args.network_weights)
    for weight in network_weights:
        weight_dir = os.path.join(args.network_weights, weight)
        parent, network_dir = os.path.split(weight_dir)
        model_name = os.path.splitext(network_dir)[0]
        if 'last' not in model_name:
            model_epoch = int(model_name.split('-')[-1])
        else:
            model_epoch = 'last'

        test_lora_dir = os.path.join(test_set_output_dir, f'lora_{model_epoch}')
        os.makedirs(test_lora_dir, exist_ok=True)

        print(f' \n step 2. make stable diffusion model')
        device = accelerator.device
        print(f' (2.1) tokenizer')
        tokenizer = train_util.load_tokenizer(args)
        print(f' (2.2) SD')
        invers_text_encoder, vae, invers_unet, load_stable_diffusion_format = train_util._load_target_model(args,
                                                                                                            weight_dtype,
                                                                                                            device,
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
            invers_unet, invers_t_enc1, invers_t_enc2 = invers_unet.to(device), invers_text_encoders[0].to(device), \
            invers_text_encoders[1].to(device)
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
        sys.path.append(os.path.dirname(__file__))
        network_module = importlib.import_module(args.network_module)
        net_kwargs = {}
        network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet,
                                                neuron_dropout=args.network_dropout, **net_kwargs, )
        print(f' (2.5.3) apply trained state dict')
        network.apply_to(text_encoder, unet, True, True)
        if args.network_weights is not None:
            info = network.load_weights(weight_dir)
        network.to(device)

        print(f' (2.4.+) model to accelerator device')
        controller = AttentionStore()
        register_attention_control(unet, controller)

        print(f' \n step 3. ground-truth image preparing')
        print(f' (3.1) prompt condition')
        context = init_prompt(tokenizer, text_encoder, device, args.prompt)
        uncon, con = torch.chunk(context, 2)

        print(f' (3.2) train images')
        trg_h, trg_w = args.resolution

        print(f' (3.3) test images')
        test_img_folder = os.path.join(args.concept_image_folder, 'test_ex/rgb')
        test_mask_folder = os.path.join(args.concept_image_folder, 'test_ex/gt')

        classes = os.listdir(test_img_folder)
        records = []
        first_elem = ['class', 'img_name']
        total_dict = {}
        for k, class_name in enumerate(classes):

            if '_' in class_name:
                trg_prompt = class_name.split('_')[-1]
            else:
                trg_prompt = class_name
            class_base_folder = os.path.join(test_lora_dir, class_name)
            os.makedirs(class_base_folder, exist_ok=True)

            image_folder = os.path.join(test_img_folder, class_name)
            mask_folder = os.path.join(test_mask_folder, class_name)
            invers_context = init_prompt(tokenizer, invers_text_encoder, device, f'a photo of {class_name}')
            inv_unc, inv_c = invers_context.chunk(2)
            test_images = os.listdir(image_folder)

            for j, test_image in enumerate(test_images):

                name, ext = os.path.splitext(test_image)

                trg_img_output_dir = os.path.join(class_base_folder, f'{name}')
                os.makedirs(trg_img_output_dir, exist_ok=True)

                test_img_dir = os.path.join(image_folder, test_image)
                Image.open(test_img_dir).convert('RGB').resize((int(trg_h),int(trg_w))).save(os.path.join(trg_img_output_dir, test_image))

                mask_img_dir = os.path.join(mask_folder, test_image)
                Image.open(mask_img_dir).convert('L').resize((int(trg_h),int(trg_w))).save(os.path.join(trg_img_output_dir, f'{name}_mask{ext}'))


                mask_np = load_image(mask_img_dir, trg_h=int(trg_h), trg_w=int(trg_w))
                mask_np = np.where(mask_np > 100, 1, 0)  # binary mask


                with torch.no_grad():
                    print(f' (1) img to latent')
                    org_img = load_image(test_img_dir, 512, 512)
                    org_vae_latent = image2latent(org_img, vae, device, weight_dtype)

                    inf_time = inference_times.tolist()
                    inf_time.reverse()  # [0,20,40,60,80,100 , ... 980]

                    latent_dict = {}
                    latent = org_vae_latent

                    t, next_time = inf_time[0], inf_time[1]
                    latent_dict[int(t)] = latent

                    noise_pred = call_unet(unet, latent, t, con[:,:3,:], [[1]], None)
                    attn_stores = controller.step_store
                    controller.reset()
                    attn_dict = {}
                    score_dict = {}
                    for layer_name in attn_stores :
                        attn = attn_stores[layer_name][0].squeeze() # head, pix_num
                        res = int(attn.shape[1] ** 0.5)
                        if 'down' in layer_name: position = 'down'
                        elif 'up' in layer_name: position = 'up'
                        else: position = 'middle'

                        if 'attentions_0' in layer_name : part = 'attn_0'
                        elif 'attentions_1' in layer_name : part = 'attn_1'
                        else : part = 'attn_2'
                        title_name = f'res_{res}_{position}_{part}'

                        # ----------------------------------------- get attn map ----------------------------------------- #
                        cls_score, normal_score, pad_score = attn.chunk(3, dim=-1) # head, pix_num
                        h = cls_score.shape[0]
                        normal_score = normal_score.unsqueeze(-1).reshape(h, res, res)
                        singl_head_normal_score = normal_score.mean(dim=0)
                        n_score = singl_head_normal_score.detach().cpu()
                        n_score = n_score / n_score.max()

                        # [1] resizing for recording
                        score_np = np.array((n_score.cpu()) * 255).astype(np.uint8)
                        mask_img = Image.open(mask_img_dir).convert("L").resize((res, res), Image.BICUBIC)
                        mask_np = np.where( (np.array(mask_img, np.uint8)) > 100, 1, 0)  # [res,res]
                        """ anormal portion score """
                        score_dict[title_name] = score_np * mask_np

                        # [2] saving n_score map
                        n_score_np = np.array((n_score.cpu()) * 255).astype(np.uint8)
                        n_score_pil = Image.fromarray(n_score_np).resize((512, 512),Image.BILINEAR)
                        n_score_pil.save(os.path.join(trg_img_output_dir,
                                                      f'{title_name}_res_{res}.png'))

                        if 'down' in layer_name : key_name = f'down_{res}'
                        elif 'up' in layer_name : key_name = f'up_{res}'
                        else : key_name = f'mid_{res}'

                        if key_name not in attn_dict :
                            attn_dict[key_name] = []
                        attn_dict[key_name].append(attn)

                    for key_name in attn_dict :
                        attn_list = attn_dict[key_name]
                        attn = torch.cat(attn_list, dim=0)
                        cls_score, n_score, pad_score = attn.chunk(3, dim=-1)
                        res = int(attn.shape[1] ** 0.5)
                        h = cls_score.shape[0]
                        n_score = n_score.unsqueeze(-1).reshape(h, res, res)
                        singl_head_normal_score = n_score.mean(dim=0)
                        n_score = singl_head_normal_score.detach().cpu()
                        n_score = np.array(((n_score / n_score.max()).cpu()) * 255).astype(np.uint8)
                        n_score_pil = Image.fromarray(n_score).resize((512, 512), Image.BILINEAR)
                        n_score_pil.save(os.path.join(trg_img_output_dir,
                                                      f'normal_{key_name}.png'))

                # ----------------------------------------------------------------------------------------------------- #
                if 'good' not in trg_prompt :
                    record = [trg_prompt, test_image]

                for k in score_dict.keys():
                    if 'good' not in trg_prompt:
                        trigger_score = score_dict[k]
                        if k not in total_dict.keys() :
                            total_dict[k] = trigger_score.sum().item()
                        else :
                            total_dict[k] += trigger_score.sum().item()
                        record.append(trigger_score.sum().item())

                    if j == 0 and k == 0 :
                        first_elem.append(k)

                if j == 0 and k == 0 :
                    records.append(first_elem)

                if 'good' not in trg_prompt:
                    records.append(record)
        total_elem = ['','']
        for k in total_dict.keys() :
            total_elem.append(total_dict[k])
        record_csv_dir = os.path.join(test_set_record_dir, f'score_epoch_{model_epoch}.csv')
        with open(record_csv_dir, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerows(records)
    # ------------------------------------------------ train check -------------------------------------------------------- #
    print(f" [2] Train")
    test_set_output_dir = os.path.join(args.output_dir, 'train_set')
    os.makedirs(test_set_output_dir, exist_ok=True)
    test_set_record_dir = os.path.join(record_output_dir, 'train_set')
    os.makedirs(test_set_record_dir, exist_ok=True)

    network_weights = os.listdir(args.network_weights)
    for weight in network_weights:
        weight_dir = os.path.join(args.network_weights, weight)
        parent, network_dir = os.path.split(weight_dir)
        model_name = os.path.splitext(network_dir)[0]
        if 'last' not in model_name:
            model_epoch = int(model_name.split('-')[-1])
        else:
            model_epoch = 'last'

        test_lora_dir = os.path.join(test_set_output_dir, f'lora_{model_epoch}')
        os.makedirs(test_lora_dir, exist_ok=True)

        print(f' \n step 2. make stable diffusion model')
        device = accelerator.device
        print(f' (2.1) tokenizer')
        tokenizer = train_util.load_tokenizer(args)
        print(f' (2.2) SD')
        invers_text_encoder, vae, invers_unet, load_stable_diffusion_format = train_util._load_target_model(args,
                                                                                                            weight_dtype,
                                                                                                            device,
                                                                                                            unet_use_linear_projection_in_v2=False, )
        invers_text_encoders = invers_text_encoder if isinstance(invers_text_encoder, list) else [invers_text_encoder]
        text_encoder, vae, unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype,
                                                                                              device,
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
            invers_unet, invers_t_enc1, invers_t_enc2 = invers_unet.to(device), invers_text_encoders[0].to(device), \
                invers_text_encoders[1].to(device)
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
        sys.path.append(os.path.dirname(__file__))
        network_module = importlib.import_module(args.network_module)
        net_kwargs = {}
        network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet,
                                                neuron_dropout=args.network_dropout, **net_kwargs, )
        print(f' (2.5.3) apply trained state dict')
        network.apply_to(text_encoder, unet, True, True)
        if args.network_weights is not None:
            info = network.load_weights(weight_dir)
        network.to(device)

        print(f' (2.4.+) model to accelerator device')
        controller = AttentionStore()
        register_attention_control(unet, controller)

        print(f' \n step 3. ground-truth image preparing')
        print(f' (3.1) prompt condition')
        context = init_prompt(tokenizer, text_encoder, device, args.prompt)
        uncon, con = torch.chunk(context, 2)

        print(f' (3.2) train images')
        trg_h, trg_w = args.resolution

        print(f' (3.3) test images')
        test_img_folder = os.path.join(args.concept_image_folder, 'train_ex/rgb')
        test_mask_folder = os.path.join(args.concept_image_folder, 'train_ex/gt')

        classes = os.listdir(test_img_folder)
        records = []
        first_elem = ['class', 'img_name']
        total_dict = {}
        for k, class_name in enumerate(classes):

            if '_' in class_name:
                trg_prompt = class_name.split('_')[-1]
            else:
                trg_prompt = class_name
            class_base_folder = os.path.join(test_lora_dir, class_name)
            os.makedirs(class_base_folder, exist_ok=True)

            image_folder = os.path.join(test_img_folder, class_name)
            mask_folder = os.path.join(test_mask_folder, class_name)
            invers_context = init_prompt(tokenizer, invers_text_encoder, device, f'a photo of {class_name}')
            inv_unc, inv_c = invers_context.chunk(2)
            test_images = os.listdir(image_folder)

            for j, test_image in enumerate(test_images):

                name, ext = os.path.splitext(test_image)

                trg_img_output_dir = os.path.join(class_base_folder, f'{name}')
                os.makedirs(trg_img_output_dir, exist_ok=True)

                test_img_dir = os.path.join(image_folder, test_image)
                Image.open(test_img_dir).convert('RGB').resize((int(trg_h), int(trg_w))).save(
                    os.path.join(trg_img_output_dir, test_image))

                mask_img_dir = os.path.join(mask_folder, test_image)
                Image.open(mask_img_dir).convert('L').resize((int(trg_h), int(trg_w))).save(
                    os.path.join(trg_img_output_dir, f'{name}_mask{ext}'))

                mask_np = load_image(mask_img_dir, trg_h=int(trg_h), trg_w=int(trg_w))
                mask_np = np.where(mask_np > 100, 1, 0)  # binary mask

                with torch.no_grad():
                    print(f' (1) img to latent')
                    org_img = load_image(test_img_dir, 512, 512)
                    org_vae_latent = image2latent(org_img, vae, device, weight_dtype)

                    inf_time = inference_times.tolist()
                    inf_time.reverse()  # [0,20,40,60,80,100 , ... 980]

                    latent_dict = {}
                    latent = org_vae_latent

                    t, next_time = inf_time[0], inf_time[1]
                    latent_dict[int(t)] = latent

                    noise_pred = call_unet(unet, latent, t, con[:, :3, :], [[1]], None)
                    attn_stores = controller.step_store
                    controller.reset()
                    attn_dict = {}
                    score_dict = {}
                    for layer_name in attn_stores:
                        attn = attn_stores[layer_name][0].squeeze()  # head, pix_num
                        res = int(attn.shape[1] ** 0.5)
                        if 'down' in layer_name:
                            position = 'down'
                        elif 'up' in layer_name:
                            position = 'up'
                        else:
                            position = 'middle'
                        if res in args.cross_map_res and position in args.trg_position:
                            if 'attentions_0' in layer_name:
                                part = 'attn_0'
                            elif 'attentions_1' in layer_name:
                                part = 'attn_1'
                            else:
                                part = 'attn_2'
                            title_name = f'res_{res}_{position}_{part}'

                            # ----------------------------------------- get attn map ----------------------------------------- #
                            cls_score, normal_score, pad_score = attn.chunk(3, dim=-1)  # head, pix_num
                            h = cls_score.shape[0]
                            normal_score = normal_score.unsqueeze(-1).reshape(h, res, res)
                            singl_head_normal_score = normal_score.mean(dim=0)
                            n_score = singl_head_normal_score.detach().cpu()
                            n_score = n_score / n_score.max()

                            # [1] resizing for recording
                            score_np = np.array((n_score.cpu()) * 255).astype(np.uint8)
                            mask_img = Image.open(mask_img_dir).convert("L").resize((res, res), Image.BICUBIC)
                            mask_np = np.where((np.array(mask_img, np.uint8)) > 100, 1, 0)  # [res,res]
                            """ anormal portion score """
                            score_dict[title_name] = score_np * mask_np

                            # [2] saving n_score map
                            n_score_np = np.array((n_score.cpu()) * 255).astype(np.uint8)
                            n_score_pil = Image.fromarray(n_score_np).resize((512, 512), Image.BILINEAR)
                            n_score_pil.save(os.path.join(trg_img_output_dir,
                                                          f'{title_name}_res_{res}.png'))

                            if 'down' in layer_name:
                                key_name = f'down_{res}'
                            elif 'up' in layer_name:
                                key_name = f'up_{res}'
                            else:
                                key_name = f'mid_{res}'

                            if key_name not in attn_dict:
                                attn_dict[key_name] = []
                            attn_dict[key_name].append(attn)

                    for key_name in attn_dict:
                        attn_list = attn_dict[key_name]
                        attn = torch.cat(attn_list, dim=0)
                        cls_score, n_score, pad_score = attn.chunk(3, dim=-1)
                        res = int(attn.shape[1] ** 0.5)
                        h = cls_score.shape[0]
                        n_score = n_score.unsqueeze(-1).reshape(h, res, res)
                        singl_head_normal_score = n_score.mean(dim=0)
                        n_score = singl_head_normal_score.detach().cpu()
                        n_score = np.array(((n_score / n_score.max()).cpu()) * 255).astype(np.uint8)
                        n_score_pil = Image.fromarray(n_score).resize((512, 512), Image.BILINEAR)
                        n_score_pil.save(os.path.join(trg_img_output_dir,
                                                      f'normal_{key_name}.png'))

                # ----------------------------------------------------------------------------------------------------- #
                if 'good' not in trg_prompt:
                    record = [trg_prompt, test_image]

                for k in score_dict.keys():
                    if 'good' not in trg_prompt:
                        trigger_score = score_dict[k]
                        if k not in total_dict.keys():
                            total_dict[k] = trigger_score.sum().item()
                        else:
                            total_dict[k] += trigger_score.sum().item()
                        record.append(trigger_score.sum().item())

                    if j == 0 and k == 0:
                        first_elem.append(k)

                if j == 0 and k == 0:
                    records.append(first_elem)

                if 'good' not in trg_prompt:
                    records.append(record)
        total_elem = ['', '']
        for k in total_dict.keys():
            total_elem.append(total_dict[k])
        record_csv_dir = os.path.join(test_set_record_dir, f'score_epoch_{model_epoch}.csv')
        with open(record_csv_dir, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerows(records)

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
    parser.add_argument("--mask_thredhold", type=float, default = 0.5)
    parser.add_argument("--inner_iteration", type=int, default=10)
    parser.add_argument("--other_token_preserving", action = 'store_true')
    parser.add_argument('--train_down', nargs='+', type=int, help='use which res layers in U-Net down', default=[])
    parser.add_argument('--train_mid', nargs='+', type=int, help='use which res layers in U-Net mid', default=[8])
    parser.add_argument('--train_up', nargs='+', type=int, help='use which res layers in U-Net up', default=[16,32,64])
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument('--trg_position', type=arg_as_list, default=['down', 'up'])
    parser.add_argument("--cross_map_res", type=arg_as_list, default=[64, 32, 16, 8])
    parser.add_argument("--detail_64", action="store_true", )


    args = parser.parse_args()
    main(args)