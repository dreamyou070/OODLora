import importlib, argparse, gc, math, os, sys, random, time, json, toml, shutil
import numpy as np
from PIL import Image
from multiprocessing import Value
from tqdm import tqdm
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import model_util
import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (ConfigSanitizer, BlueprintGenerator, )
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import prepare_scheduler_for_custom_training
import torch
from torch import nn
import wandb
from attention_store import AttentionStore
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
            if is_cross_attention and trg_indexs_list is not None and ('up' in layer_name or 'mid' in layer_name) :
                good_map = attention_probs[:, :,1]
                bad_map = attention_probs[:,:,2] # [batch*head, pixel_num, 1]
                attn_score_map = torch.cat([good_map, bad_map], dim=-1)
                controller.store(attn_score_map, layer_name)
                """
                batch, pix_num, _ = query.shape
                res = int(pix_num) ** 0.5
                if res in args.cross_map_res :
                    batch_num = len(trg_indexs_list)
                    attention_probs_batch = torch.chunk(attention_probs, batch_num, dim=0)
                    attn_list = []
                    
                    for batch_idx, attn_probs in enumerate(attention_probs_batch):
                        
                        #batch_trg_index = trg_indexs_list[batch_idx]  # two times
                        good_map = attn_probs[:,:,1]
                        bad_map = attn_probs[:,:,2]
                        if good_map.dim() == 2 :
                            good_map = good_map.unsqueeze(-1)
                            bad_map = bad_map.unsqueeze(-1)
                        attn_score_map = torch.cat([good_map, bad_map], dim=-1)
                        attn_list.append(attn_score_map)
                    batch_attn_map = torch.cat(attn_list, dim=0)
                    controller.store(batch_attn_map, layer_name)
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


class NetworkTrainer:

    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    def generate_step_logs(self, loss_dict, lr_scheduler, keys_scaled=None, mean_norm=None, maximum_norm=None,
                           **kwargs):
        logs = {}
        for k, v in loss_dict.items():
            logs[k] = v
        # ------------------------------------------------------------------------------------------------------------------------------
        # updating kwargs with new loss logs ...
        if kwargs is not None:
            logs.update(kwargs)
        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm
        lrs = lr_scheduler.get_last_lr()
        if args.network_train_text_encoder_only or len(lrs) <= 2:  # not block lr (or single block)
            if args.network_train_unet_only:
                logs["lr/unet"] = float(lrs[0])
            elif args.network_train_text_encoder_only:
                logs["lr/textencoder"] = float(lrs[0])
            else:
                logs["lr/textencoder"] = float(lrs[0])
                logs["lr/unet"] = float(lrs[-1])  # may be same to textencoder

            if (
                    args.optimizer_type.lower().startswith(
                        "DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                        lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0][
                    "lr"])
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith(
                        "DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                            lr_scheduler.optimizers[-1].param_groups[i]["d"] *
                            lr_scheduler.optimizers[-1].param_groups[i]["lr"])

        return logs


    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet


    def extract_triggerword_index(self, input_ids):
        cls_token = 49406
        pad_token = 49407
        if input_ids.dim() == 3:
            input_ids = torch.flatten(input_ids, start_dim=1)
        batch_num, sen_len = input_ids.size()
        batch_index_list = []

        for batch_index in range(batch_num):
            token_ids = input_ids[batch_index, :].squeeze()
            index_list = []
            for index, token_id in enumerate(token_ids):
                if token_id != cls_token and token_id != pad_token:
                    index_list.append(index)
            batch_index_list.append(index_list)
        return batch_index_list

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)  # batch, torch_num, sen_len
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids,
                                                             tokenizers[0], text_encoders[0],
                                                             weight_dtype)
        return encoder_hidden_states

    def get_class_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["class_input_ids"].to(accelerator.device)  # batch, torch_num, sen_len
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids,
                                                             tokenizers[0], text_encoders[0],
                                                             weight_dtype)
        return encoder_hidden_states

    def call_unet(self,
                  args, accelerator, unet,
                  noisy_latents, timesteps,
                  text_conds, batch, weight_dtype,
                  trg_indexs_list,
                  mask_imgs):
        noise_pred = unet(noisy_latents,
                          timesteps,
                          text_conds,
                          trg_indexs_list=trg_indexs_list,
                          mask_imgs=mask_imgs, ).sample
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def train(self, args):

        args.logging_dir = os.path.join(args.output_dir, 'logs')
        parent, name = os.path.split(args.output_dir)
        args.wandb_run_name = name

        print(f'\n step 1. setting')
        print(f' (1) session')
        if args.process_title: setproctitle(args.process_title)
        else: setproctitle('parksooyeon')
        print(f' (2) seed')
        if args.seed is None: args.seed = random.randint(0, 2 ** 32)
        set_seed(args.seed)

        print(f'\n step 2. dataset')
        tokenizer = train_util.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
        train_util.prepare_dataset_args(args, True)
        use_class_caption = args.class_caption is not None

        print(f' (2.1) training dataset')
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
            user_config = {}
            user_config['datasets'] = [{"subsets": None}]
            subsets_dict_list = []
            for subsets_dict in config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir, args.class_caption):
                if use_class_caption :
                    subsets_dict['class_caption'] = args.class_caption
                subsets_dict_list.append(subsets_dict)
                user_config['datasets'][0]['subsets'] = subsets_dict_list
            print(f'User config: {user_config}')


            # --------------------------------------------------------------------------------------------------------
            """ config_util.generate_dreambooth_subsets_config_by_subdirs """
            blueprint = blueprint_generator.generate(user_config, # about data directory
                                                     args,
                                                     tokenizer=tokenizer)

            print(f'blueprint.dataset_group : {blueprint.dataset_group}')
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

        else:
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)


        print(f' (2.3) collater')
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)


        print(f'\n step 3. preparing accelerator')
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process
        if args.log_with == 'wandb' and is_main_process:
            wandb.init(project=args.wandb_init_name, name=args.wandb_run_name)

        print(f'\n step 4. save directory')
        save_base_dir = args.output_dir
        _, folder_name = os.path.split(save_base_dir)
        record_save_dir = os.path.join(args.output_dir, "record")
        os.makedirs(record_save_dir, exist_ok=True)
        print(f' (4.1) config saving')
        with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        #logging_file = os.path.join(args.output_dir, f"validation_log_{time}.txt")

        print(f'\n step 5. model')
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
        print(f' (5.1) base model')
        model_version, enc_text_encoder, enc_vae, enc_unet = self.load_target_model(args, weight_dtype, accelerator)
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        enc_text_encoders = enc_text_encoder if isinstance(enc_text_encoder, list) else [enc_text_encoder]
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0" : vae.set_use_memory_efficient_attention_xformers(args.xformers)
        print(' (5.2) lora model')
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae,
                                                    text_encoder, unet, neuron_dropout=args.network_dropout, **net_kwargs, )
        if network is None:
            return
        print(' (5.3) lora with unet and text encoder')
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = not args.network_train_unet_only
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)
        print(' (5.4) lora resume?')
        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")
        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for t_enc in text_encoders:
                t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        print(f'\n step 6. optimizer')
        try:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        except :
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)
        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        print(f' step 7. dataloader')
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)
        train_dataloader = torch.utils.data.DataLoader(train_dataset_group, batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       collate_fn=collater, num_workers=n_workers,
                                                       persistent_workers=args.persistent_data_loader_workers, )
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
            accelerator.print( f"override steps. steps for {args.max_train_epochs} epochs / {args.max_train_steps}")

        print(f'\n step 7. lr')
        lr_scheduler = train_util.get_scheduler_fix(args,optimizer, accelerator.num_processes)
        if args.full_fp16:
            assert (args.mixed_precision == "fp16"), "full_fp16 requires mixed precision='fp16'"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (args.mixed_precision == "bf16"), "full_bf16 requires mixed precision='bf16' / mixed_precision='bf16'"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet.requires_grad_(False)
        unet.to(dtype=weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)
        enc_unet.requires_grad_(False)
        enc_unet.to(dtype=weight_dtype)
        for enc_t_enc in enc_text_encoders:
            enc_t_enc.requires_grad_(False)

        print(f'\n step 7. training preparing')
        if train_unet and train_text_encoder:
            if len(text_encoders) > 1:
                unet, t_enc1, t_enc2, network, optimizer, train_dataloader,lr_scheduler, = accelerator.prepare(
                    unet, text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler,)
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
                enc_t_enc1, enc_t_enc2, enc_unet, = accelerator.prepare(enc_text_encoders[0], enc_text_encoders[1], enc_unet)
                enc_text_encoder = enc_text_encoders = [enc_t_enc1, enc_t_enc2]
                del enc_t_enc1, enc_t_enc2
            else:
                unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler)
                text_encoders = [text_encoder]
                enc_t_enc, enc_unet, = accelerator.prepare(enc_text_encoder, enc_unet)
                enc_text_encoders = [enc_text_encoder]
        elif train_unet:
            unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, network, optimizer, train_dataloader, lr_scheduler)
            enc_t_enc, enc_unet, = accelerator.prepare(enc_text_encoder, enc_unet)
        elif train_text_encoder:
            if len(text_encoders) > 1:
                t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler)
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
                enc_t_enc1, enc_t_enc2, enc_unet, = accelerator.prepare(enc_text_encoders[0], enc_text_encoders[1], enc_unet)
                enc_text_encoder = enc_text_encoders = [enc_t_enc1, enc_t_enc2]
                del enc_t_enc1, enc_t_enc2
            else:
                text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    text_encoder, network, optimizer, train_dataloader, lr_scheduler)
                text_encoders = [text_encoder]
                enc_t_enc, enc_unet, = accelerator.prepare(enc_text_encoder, enc_unet)
                enc_text_encoders = [enc_text_encoder]
            unet.to(accelerator.device,dtype=weight_dtype)  # move to device because unet is not prepared by accelerator
            enc_unet.to(accelerator.device,dtype=weight_dtype)  # move to device because unet is not prepared by accelerator
        else:
            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)
        text_encoders = train_util.transform_models_if_DDP(text_encoders)
        unet, network = train_util.transform_models_if_DDP([unet, network])
        enc_text_encoders = train_util.transform_models_if_DDP(enc_text_encoders)
        enc_unet = train_util.transform_models_if_DDP([enc_unet])[0]
        if args.gradient_checkpointing:
            unet.train()
            for t_enc in text_encoders:
                t_enc.train()
                if train_text_encoder:
                    t_enc.text_model.embeddings.requires_grad_(True)
            if not train_text_encoder:  # train U-Net only
                unet.parameters().__next__().requires_grad_(True)
        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()
            enc_unet.eval()
            for enc_t_enc in enc_text_encoders:
                enc_t_enc.eval()
        del t_enc
        del enc_text_encoders, enc_vae

        network.prepare_grad_etc(text_encoder, unet)
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

        print(f'\n step 7. training preparing')
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        print(f'\n step 8. training')
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
        attention_storer = AttentionStore()
        register_attention_control(unet, attention_storer, mask_threshold=args.mask_threshold)

        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
        global_step = 0
        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=1000, clip_sample=False)
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
        if accelerator.is_main_process:
            init_kwargs = {}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers("network_train" if args.log_tracker_name is None else args.log_tracker_name,
                                      init_kwargs=init_kwargs)

        loss_list = []
        loss_total = 0.0
        del train_dataset_group
        # callback for step start
        if hasattr(network, "on_step_start"):
            on_step_start = network.on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None

        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)
            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata_to_save = {}
            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            metadata_to_save.update(sai_metadata)
            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        record_file = os.path.join(args.output_dir, 'score_record.txt')
        for epoch in range(args.start_epoch, args.start_epoch+num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1
            network.on_epoch_start(text_encoder, unet)

            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step
                with accelerator.accumulate(network):
                    on_step_start(text_encoder, unet)
                    with torch.no_grad():
                        latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample()
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                        latents = latents * self.vae_scale_factor

                    train_class_list = batch["train_class_list"]
                    train_indexs = [index for index, i in enumerate(train_class_list) if i == 1]
                    test_indexs = [index for index, i in enumerate(train_class_list) if i != 1]
                    trg_indexs = batch["trg_indexs_list"]
                    log_loss = {}

                    # -------------------------------------------------- (1) attention loss -------------------------------------------------- #

                    total_loss = 0
                    with torch.set_grad_enabled(train_text_encoder):
                        """ text = CLS bad good EOS """
                        text_encoder_conds = self.get_text_cond(args, accelerator, batch, tokenizers, text_encoders, weight_dtype)
                        text_encoder_conds = text_encoder_conds[:,:4,:] # add one pad token (EOS token)
                    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)
                    with accelerator.autocast():
                        self.call_unet(args, accelerator, unet, noisy_latents, timesteps, text_encoder_conds,
                                       batch, weight_dtype, trg_indexs, test_indexs)
                        attn_dict = attention_storer.step_store
                        attention_storer.reset()
                        normal_loss, anormal_loss, cross_loss = 0, 0, 0
                        img_masks = batch["img_masks"].to(accelerator.device)      # [Batch, 1, 512, 512], foreground = white = 1, background = black = 0
                        binary_map = batch['anormal_masks'].to(accelerator.device) # [Batch, 1, 512, 512], normal = black = 0, anormal = white = 1
                        batch_num = img_masks.shape[0]

                        for layer in attn_dict.keys():
                            attn_score = attn_dict[layer][0]                                               # [batch*head, pixel_num, 2]
                            normal_score_map, anormal_score_map = torch.chunk(attn_score, 2, dim=-1)       # [batch*head, pixel_num, 1]
                            if normal_score_map.dim() != 3:
                                normal_score_map = normal_score_map.unsqueeze(-1)
                                anormal_score_map = anormal_score_map.unsqueeze(-1)
                            pix_num = normal_score_map.shape[1]
                            res = int(math.sqrt(pix_num))
                            if res in args.cross_map_res :

                                from torchvision import transforms
                                resize_transform = transforms.Resize((res, res))

                                img_masks_res = (1 -(resize_transform(img_masks) == 0.0).float())  # background = 0, foreground = 1
                                binary_map_res = (1-(resize_transform(binary_map) == 0.0).float()) # normal = 0, anormal = 1
                                normal_mask_res = img_masks_res * (1-binary_map_res) # [1,1,res,res]
                                anormal_mask_res = img_masks_res * binary_map_res     # [1,1,res,res]

                                normal_score_map_batch = torch.chunk(normal_score_map,  batch_num, dim=0)  # batch*head, pixel_num, 1
                                anormal_score_map_batch = torch.chunk(anormal_score_map, batch_num, dim=0) # batch*head, pixel_num, 1

                                for i in range(batch_num):
                                    normal_score_map_i =normal_score_map_batch[i].reshape(8, res, res, -1).squeeze(-1)    # [h, res, res]
                                    anormal_score_map_i = anormal_score_map_batch[i].reshape(8, res, res, -1).squeeze(-1) # [h, res, res]
                                    b, H, W, = normal_score_map_i.shape

                                    # -------------------------------------------------- (1-1) normal loss -------------------------------------------------- #

                                    normal_mask_ = normal_mask_res[i, :, :].repeat(8, 1, 1) # [h, res, res] # """ background is zero """
                                    anormal_mask_ = anormal_mask_res[i, :, :].repeat(8, 1, 1) # [h, res, res]

                                    normal_total_score = normal_score_map_i.reshape(b, -1).sum(dim=-1)
                                    anormal_total_score = anormal_score_map_i.reshape(b, -1).sum(dim=-1) # [head, res*res]

                                    normal_pos_normal_score =     (normal_mask_ * normal_score_map_i).reshape(b, -1).sum(dim=-1) # [8, res,res] * [8, res,res]
                                    anormal_pos_anormal_score = (anormal_mask_ * anormal_score_map_i).reshape(b, -1).sum(dim=-1)

                                    normal_activation_value = normal_pos_normal_score / normal_total_score
                                    anormal_activation_value = anormal_pos_anormal_score / anormal_total_score
                                    normal_loss += (1.0 - torch.mean(normal_activation_value)) ** 2
                                    if len(test_indexs) > 0 :
                                        print(f'test_indexs : {test_indexs}')

                                        anormal_loss += (1.0 - torch.mean(anormal_activation_value)) ** 2
                                        # -------------------------------------------------- (2-1) normal and anormal position ------------------------------------ #
                                        # binary_aug_tensor = 8,32,321
                                        # img_mask = 8,32,32,1
                                        score_map = torch.cat([normal_score_map_i.unsqueeze(-1), anormal_score_map_i.unsqueeze(-1)], dim=-1).softmax(dim=-1)  #
                                        flatten_score_map = score_map.view(-1, 2)

                                        position_map = torch.cat([normal_mask_.unsqueeze(-1),anormal_mask_.unsqueeze(-1)], dim=-1)
                                        position_map = position_map.view(-1, 2)

                                        normal_pairs, anormal_pairs, score_pairs = [], [], []
                                        normal_answers, anormal_answers, answers = [], [], []

                                        normal_num, anormal_num = 0, 0
                                        for i in range(flatten_score_map.shape[0]):

                                            position_info = position_map[i]

                                            if position_info[0] == 1 and position_info[1] != 1 : # normal_pixel
                                                normal_score_pair = flatten_score_map[i] # normal = 0, anormal = 1
                                                normal_pairs.append(normal_score_pair)
                                                normal_answers.append(1-position_info)
                                                normal_num += 1

                                            elif position_info[1] == 1 and position_info[0] != 1: # anormal_pixel
                                                anormal_score_pair = flatten_score_map[i] # normal = 0, anormal = 1
                                                anormal_pairs.append(anormal_score_pair)
                                                anormal_answers.append(1 - position_info)
                                                anormal_num += 1

                                            if position_info[0] == 1 or position_info[1] == 1:
                                                score_pair = flatten_score_map[i] # normal = 0, anormal = 1
                                                answer = 1-position_info
                                                score_pairs.append(score_pair)
                                                answers.append(answer)

                                        if len(normal_pairs) == 0 or len(anormal_pairs) == 0 :
                                            print(f'normal_num : {normal_num}, anormal_num : {anormal_num} '
                                                  f'| anormal_mask_ : {anormal_mask_.sum()} | anormal_mask_res : {anormal_mask_res.sum()} '
                                                  f'| binary_map_res : {binary_map_res.sum()} '
                                                  f'| binary_map : {binary_map.sum()} '
                                                  f'| binary_map : {binary_map.shape} ')
                                            print(f'wronge, sum of normal_mask_res : {torch.sum(normal_mask_res)}')
                                            print(f'wronge, sum of anormal_mask_res : {torch.sum(anormal_mask_res)}')

                                        normal_pairs = torch.stack(normal_pairs)
                                        anormal_pairs = torch.stack(anormal_pairs)
                                        score_pairs = torch.stack(score_pairs)

                                        normal_answers = torch.stack(normal_answers)
                                        anormal_answers = torch.stack(anormal_answers)
                                        answers = torch.stack(answers)

                                        cross_ent_loss = torch.nn.BCELoss()(score_pairs, answers)
                                        cross_loss += cross_ent_loss.mean()

                                        normal_cross_loss = torch.nn.BCELoss()(normal_pairs,normal_answers.long().to(accelerator.device, dtype=weight_dtype))
                                        anormal_cross_loss = torch.nn.BCELoss()(anormal_pairs,anormal_answers.long().to(accelerator.device, dtype=weight_dtype))

                                        log_loss["loss/normal_cross_loss"] = normal_cross_loss.mean().item()
                                        log_loss["loss/anormal_cross_loss"] = anormal_cross_loss.mean().item()


                                    #normal_position_normal_score +=
                        log_loss["loss/normal_pixel_reverse_normal_loss"] = normal_loss.mean().item()

                        if len(test_indexs) > 0:
                            log_loss["loss/anormal_pixel_reverse_anormal_loss"] = anormal_loss.mean().item()
                            log_loss["loss/cross_entropy_loss"] = cross_loss.mean().item()

                        #record = {"normal_pixel_reverse_normal_score": normal_loss.mean().item(),
                        #          "anormal_pixel_reverse_anormal_score": anormal_loss.mean().item(),}
                        #with open(record_file, 'a') as f:
                        #    f.write(json.dumps(record) + '\n')

                        # attn_loss = normal_loss.mean() + anormal_loss.mean()+ cross_loss.mean()
                        # attn_loss = cross_loss.mean()
                        if len(test_indexs) > 0:
                            attn_loss = normal_loss.mean() + anormal_loss.mean() + cross_loss.mean()
                        else :
                            attn_loss = normal_loss.mean()

                    total_loss += attn_loss

                    # ---------------------------------------------------------------------------------------------------------------------
                    # (3.3) natural training
                    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args,
                                                                                                       noise_scheduler,
                                                                                                       latents)
                    with accelerator.autocast():
                        noise_pred = self.call_unet(args, accelerator, unet,
                                                    noisy_latents, timesteps,
                                                    text_encoder_conds, batch, weight_dtype,
                                                    None, mask_imgs =None)
                    attention_storer.reset()
                    if args.v_parameterization:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise
                    task_loss = torch.nn.functional.mse_loss(noise_pred.float(),
                                                             target.float(), reduction="none")
                    task_loss = task_loss.mean([1, 2, 3])  # * batch["loss_weights"]  # 各sampleごとのweight
                    task_loss = task_loss.mean()
                    log_loss["loss/task_loss"] = task_loss
                    task_loss = task_loss * args.task_loss_weight
                    total_loss += task_loss.mean()

                    # ------------------------------------------------------------------------------------
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = network.get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = network.apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device)
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    # (4) sample images
                    # self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)
                    if attention_storer is not None: attention_storer.step_store = {}
                    # (5) save or erase model
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)
                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)
                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as,
                                                                                 remove_step_no)
                                remove_model(remove_ckpt_name)
                current_loss = total_loss.detach().item()
                log_loss["loss/current_loss"] = current_loss
                # ------------------------------------------------------------------------------------------------------------------------------
                if epoch == args.start_epoch :
                    loss_list.append(current_loss)
                else:
                    loss_total -= loss_list[step]
                    loss_list[step] = current_loss
                loss_total += current_loss
                avr_loss = loss_total / len(loss_list)
                log_loss["loss/avr_loss"] = avr_loss
                # ------------------------------------------------------------------------------------------------------------------------------
                progress_bar.set_postfix(**log_loss)
                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **log_loss})
                if args.logging_dir is not None:
                    logs = self.generate_step_logs(log_loss, lr_scheduler, keys_scaled, mean_norm, maximum_norm)
                    accelerator.log(logs, step=global_step)
                if is_main_process:
                    wandb.log(logs)
                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_total / len(loss_list)}
                accelerator.log(logs, step=epoch + 1)
            accelerator.wait_for_everyone()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    print(f'model save on {ckpt_name}')
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)
                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                        remove_model(remove_ckpt_name)
                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            if epoch % args.sample_every_n_epochs == 0 and is_main_process:
                self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer,
                                   text_encoder, unet)
            if attention_storer is not None:
                attention_storer.reset()

        if is_main_process:
            network = accelerator.unwrap_model(network)
        accelerator.end_training()
        if is_main_process and args.save_state:
            train_util.save_state_on_train_end(args, accelerator)
        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)
            print("model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. setting
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    parser.add_argument("--wandb_log_template_path", type=str)
    parser.add_argument("--wandb_key", type=str)
    # step 2. dataset
    train_util.add_dataset_arguments(parser, True, True, True)
    parser.add_argument("--mask_dir", type=str, default='')
    parser.add_argument("--class_caption", type=str, default='')
    parser.add_argument("--no_metadata", action="store_true",
                        help="do not save metadata in output model")
    # step 3. model
    train_util.add_sd_models_arguments(parser)
    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--network_dim", type=int, default=None,
                        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）")
    parser.add_argument("--network_alpha", type=float, default=1,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)", )
    parser.add_argument("--network_dropout", type=float, default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)", )
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value) / ネットワークへの追加の引数")
    # step 4. training
    train_util.add_training_arguments(parser, True)
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")
    # step 5. optimizer
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）", )
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
    parser.add_argument("--dim_from_weights", action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する", )
    parser.add_argument("--scale_weight_norms", type=float, default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. ", )
    parser.add_argument("--base_weights", type=str, default=None, nargs="*",
                        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル", )
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*",
                        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う", )

    parser.add_argument("--mask_threshold", type=float, default=0.5)

    parser.add_argument("--contrastive_eps", type=float, default=0.00005)
    parser.add_argument("--resume_lora_training", action="store_true",)
    parser.add_argument("--start_epoch", type = int, default = 0)
    parser.add_argument("--valid_data_dir", type=str)
    parser.add_argument("--task_loss_weight", type=float, default=0.5)
    parser.add_argument("--anormal_training", action = 'store_true')

    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--cross_map_res", type=arg_as_list, default=[64,32,16,8])
    parser.add_argument("--normal_training", action="store_true", )
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    trainer = NetworkTrainer()
    trainer.train(args)