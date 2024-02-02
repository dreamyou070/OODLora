import importlib, argparse, gc, math, os, sys, random, time, json, toml, shutil
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
from attention_store import AttentionStore

try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None
import os


def register_attention_control(unet: nn.Module, controller: AttentionStore, ):  # if mask_threshold is 1, use itself

    def ca_forward(self, layer_name):

        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            random_hidden_states = torch.randn_like(hidden_states)
            random_hidden_states = random_hidden_states.to(hidden_states.device)

            query = self.to_q(hidden_states) # "to_q learning"
            random_query = self.to_q(random_hidden_states)
            if args.query_add_random :
                random_query = random_query + query
            # --------------------------------------------------------------------------------------------------
            controller.save_query(query, layer_name)
            controller.save_query(random_query, layer_name)
            # --------------------------------------------------------------------------------------------------
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)

            query = self.reshape_heads_to_batch_dim(query)
            random_query = self.reshape_heads_to_batch_dim(random_query)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            if self.upcast_attention:
                query = query.float()
                random_query = random_query.float()
                key = key.float()

            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                            device=query.device), query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            random_attention_scores = torch.baddbmm(
                torch.empty(random_query.shape[0], random_query.shape[1], key.shape[1], dtype=random_query.dtype,
                            device=random_query.device), random_query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            random_attention_probs = random_attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)
            random_attention_probs = random_attention_probs.to(value.dtype)

            if is_cross_attention:
                if args.cls_training:
                    trg_map = attention_probs[:, :, :2]
                    random_trg_map = random_attention_probs[:, :, :2]
                else:
                    trg_map = attention_probs[:, :, 1]
                    random_trg_map = random_attention_probs[:, :, 1]
                controller.store(trg_map, layer_name)
                controller.store(random_trg_map, layer_name)

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

        print(f'\n step 1. setting')
        print(f' (1) session')
        if args.process_title:
            setproctitle(args.process_title)
        else:
            setproctitle('parksooyeon')
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
            for subsets_dict in config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir,
                                                                                          args.reg_data_dir,
                                                                                          args.class_caption):
                if use_class_caption:
                    subsets_dict['class_caption'] = args.class_caption
                subsets_dict_list.append(subsets_dict)
                user_config['datasets'][0]['subsets'] = subsets_dict_list
            print(f'User config: {user_config}')

            # --------------------------------------------------------------------------------------------------------
            """ config_util.generate_dreambooth_subsets_config_by_subdirs """
            blueprint = blueprint_generator.generate(user_config,  # about data directory
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

        print(f'\n step 4. save directory')
        save_base_dir = args.output_dir
        os.makedirs(save_base_dir, exist_ok=True)
        _, folder_name = os.path.split(save_base_dir)
        record_save_dir = os.path.join(args.output_dir, "record")
        os.makedirs(record_save_dir, exist_ok=True)
        print(f' (4.1) config saving')
        with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        # logging_file = os.path.join(args.output_dir, f"validation_log_{time}.txt")

        print(f'\n step 5. model')
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
        print(f' (5.1) base model')

        model_version, enc_text_encoder, enc_vae, enc_unet = self.load_target_model(args, weight_dtype, accelerator)
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        enc_text_encoders = enc_text_encoder if isinstance(enc_text_encoder, list) else [enc_text_encoder]
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0": vae.set_use_memory_efficient_attention_xformers(args.xformers)
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
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet,
                                                                    **net_kwargs)
        else:
            network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae,
                                                    text_encoder, unet, neuron_dropout=args.network_dropout,
                                                    **net_kwargs, )
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
        except:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

        print(f'\n (6.2) frozen or not')
        if args.unet_frozen :
            params = []
            unet_loras = network.unet_loras
            for unet_lora in unet_loras:
                lora_name = unet_lora.lora_name
                if 'to_k' in lora_name or 'to_v' in lora_name:
                    if 'attn2' in lora_name :
                        print(f'unet frozen layer : {lora_name}')
                        params.extend(unet_lora.parameters())
                    else :
                        unet_lora.requires_grad = False
                else :
                        unet_lora.requires_grad = False
            te_loras = network.text_encoder_loras
            for te_lora in te_loras:
                params.extend(te_lora.parameters())
            trainable_params = [{"params": params, "lr": args.unet_lr}]

        if args.text_frozen :
            params = []
            unet_loras = network.unet_loras
            for unet_lora in unet_loras:
                lora_name = unet_lora.lora_name
                if 'to_k' in lora_name or 'to_v' in lora_name:
                    if 'attn2' in lora_name:
                        unet_lora.requires_grad = False
                    else:
                        params.extend(unet_lora.parameters())
                else:
                    params.extend(unet_lora.parameters())
            te_loras = network.text_encoder_loras
            for te_lora in te_loras:
                te_lora.requires_grad = False
            trainable_params = [{"params": params, "lr": args.text_encoder_lr}]
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
            accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs / {args.max_train_steps}")

        print(f'\n step 7. lr')
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
        if args.full_fp16:
            assert (args.mixed_precision == "fp16"), "full_fp16 requires mixed precision='fp16'"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                    args.mixed_precision == "bf16"), "full_bf16 requires mixed precision='bf16' / mixed_precision='bf16'"
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
                unet, t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler, = accelerator.prepare(
                    unet, text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler, )
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
                enc_t_enc1, enc_t_enc2, enc_unet, = accelerator.prepare(enc_text_encoders[0], enc_text_encoders[1],
                                                                        enc_unet)
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
            text_encoder.to(accelerator.device)
        elif train_text_encoder:
            if len(text_encoders) > 1:
                t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler)
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
                enc_t_enc1, enc_t_enc2, enc_unet, = accelerator.prepare(enc_text_encoders[0], enc_text_encoders[1],
                                                                        enc_unet)
                enc_text_encoder = enc_text_encoders = [enc_t_enc1, enc_t_enc2]
                del enc_t_enc1, enc_t_enc2
            else:
                text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    text_encoder, network, optimizer, train_dataloader, lr_scheduler)
                text_encoders = [text_encoder]
                enc_t_enc, enc_unet, = accelerator.prepare(enc_text_encoder, enc_unet)
                enc_text_encoders = [enc_text_encoder]
            unet.to(accelerator.device,
                    dtype=weight_dtype)  # move to device because unet is not prepared by accelerator
            enc_unet.to(accelerator.device,
                        dtype=weight_dtype)  # move to device because unet is not prepared by accelerator
        else:
            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer,
                                                                                     train_dataloader, lr_scheduler)
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
        register_attention_control(unet, attention_storer)

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets.
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(
            f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,
            # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
        }

        # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
        assert (
                len(train_dataset_group.datasets) == 1
        ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

        dataset = train_dataset_group.datasets[0]

        dataset_dirs_info = {}
        reg_dataset_dirs_info = {}
        for subset in dataset.subsets:
            info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
            info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats,
                                                        "img_count": subset.img_count}

        metadata.update({
            "ss_batch_size_per_device": args.train_batch_size,
            "ss_total_batch_size": total_batch_size,
            "ss_resolution": args.resolution,
            "ss_color_aug": bool(args.color_aug),
            "ss_flip_aug": bool(args.flip_aug),
            "ss_random_crop": bool(args.random_crop),
            "ss_shuffle_caption": bool(args.shuffle_caption),
            "ss_enable_bucket": bool(dataset.enable_bucket),
            "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
            "ss_min_bucket_reso": dataset.min_bucket_reso,
            "ss_max_bucket_reso": dataset.max_bucket_reso,
            "ss_keep_tokens": args.keep_tokens,
            "ss_dataset_dirs": json.dumps(dataset_dirs_info),
            "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
            "ss_tag_frequency": json.dumps(dataset.tag_frequency),
            "ss_bucket_info": json.dumps(dataset.bucket_info), })

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process,
                            desc="steps")
        global_step = 0

        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=1000, clip_sample=False)
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)

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
            save_model_base_dir = os.path.join(args.output_dir, "models")
            os.makedirs(save_model_base_dir, exist_ok=True)
            ckpt_file = os.path.join(save_model_base_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # training loop
        if is_main_process:
            gradient_dict = {}
            loss_dict = {}

        norm = {}
        for epoch in range(args.start_epoch, args.start_epoch + num_train_epochs):

            accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + num_train_epochs}")
            current_epoch.value = epoch + 1
            network.on_epoch_start(text_encoder, unet)
            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step
                # with accelerator.accumulate(network):
                on_step_start(text_encoder, unet)
                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device)
                    else:
                        latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample()
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                    latents = latents * self.vae_scale_factor
                with torch.set_grad_enabled(train_text_encoder):
                    text_encoder_conds = self.get_text_cond(args, accelerator, batch, tokenizers, text_encoders,
                                                            weight_dtype)
                    if args.truncate_pad:
                        text_encoder_conds = text_encoder_conds[:, :args.truncate_length, :]
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args,
                                                                                                   noise_scheduler,
                                                                                                   latents)
                with accelerator.autocast():
                    noise_pred = self.call_unet(args,
                                                accelerator,
                                                unet,
                                                noisy_latents,
                                                timesteps,
                                                text_encoder_conds,
                                                batch,
                                                weight_dtype, 1, None)
                # ------------------------------------- (1) task loss ------------------------------------- #
                if args.do_task_loss:
                    if args.v_parameterization:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])
                    loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                    loss = loss * loss_weights
                    task_loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし
                    task_loss = task_loss * args.task_loss_weight

                # ------------------------------------- (2) attn loss ------------------------------------- #
                attn_dict = attention_storer.step_store
                query_dict = attention_storer.query_dict
                attention_storer.reset()
                attn_loss, dist_loss = 0, 0
                features = []
                for i, layer_name in enumerate(attn_dict.keys()):

                    map = attn_dict[layer_name][0].squeeze()  # 8, res*res, c
                    random_map = attn_dict[layer_name][1].squeeze()  # 8, res*res, c

                    if args.cls_training:
                        cls_map, score_map = torch.chunk(map, 2, dim=-1) # 8, res*res, c
                        cls_map = cls_map.squeeze()     # 8, res*res
                        score_map = score_map.squeeze() # 8, res*res

                        cls_random_map, score_random_map = torch.chunk(random_map, 2, dim=-1)
                        cls_random_map = cls_random_map.squeeze()
                        score_random_map = score_random_map.squeeze() # 8, res*res

                    else:
                        score_map = map.squeeze()  # 8, res*res
                        score_random_map = random_map.squeeze()  # 8, res*res
                    res = int(map.shape[1] ** 0.5)
                    head_num = int(score_map.shape[0])
                    do_mask_loss = False
                    if res in args.cross_map_res:
                        if 'down' in layer_name:
                            position = 'down'
                        elif 'up' in layer_name:
                            position = 'up'
                        elif 'mid' in layer_name:
                            position = 'mid'
                        if 'attentions_0' in layer_name:
                            part = 'attn_0'
                        elif 'attentions_1' in layer_name:
                            part = 'attn_1'
                        else:
                            part = 'attn_2'
                        if res == 64:
                            if args.detail_64_up:
                                if 'up' in layer_name:
                                    do_mask_loss = True
                            if args.detail_64_down:
                                if 'down' in layer_name:
                                    do_mask_loss = True
                        else:
                            if position in args.trg_position:
                                do_mask_loss = True
                        if do_mask_loss:

                            if part in args.trg_part or int(res) == 8:

                                query = query_dict[layer_name][0].squeeze()
                                random_query = query_dict[layer_name][1].squeeze()

                                pix_num = query.shape[0]  # 4096             # 4096
                                res = int(pix_num ** 0.5)  # 64

                                anormal_mask = batch["anormal_masks"][0][res].unsqueeze(
                                    0)  # [1,1,res,res], foreground = 1
                                mask = anormal_mask.squeeze()  # res,res
                                anormal_mask = torch.stack([mask.flatten() for i in range(head_num)],
                                                           dim=0)  # .unsqueeze(-1)  # 8, res*res
                                anormal_position = torch.where((anormal_mask == 0), 1, 0)  # head, pix_num
                                normal_position = 1 - anormal_position
                                back_position = 1 - normal_position
                                object_position = mask.flatten()  # pix_num

                                # (0) random features
                                random_features = []
                                for i in range(pix_num) :
                                    r_feat = random_query[i, :].squeeze()
                                    random_features.append(r_feat.unsqueeze(0))
                                random_vectors = torch.cat(random_features, dim=0)  # sample, dim

                                # (1) object features
                                for i in range(pix_num):
                                    if object_position[i] == 1:
                                        feat = query[i, :].squeeze()  # dim
                                        features.append(feat.unsqueeze(0))
                                normal_vectors = torch.cat(features, dim=0)  # sample, dim
                                """
                                if 'mean' not in norm.keys() :
                                    normal_vector_mean_torch = torch.mean(normal_vectors, dim=0)
                                    normal_vectors_cov_torch = torch.cov(normal_vectors.transpose(0, 1))
                                else :
                                    normal_vector_mean_torch = norm['mean']
                                    normal_vectors_cov_torch = norm['cov']
                                norm['mean'] = torch.mean(normal_vectors, dim=0)
                                norm['cov'] = torch.cov(normal_vectors.transpose(0, 1))
                                """
                                normal_vector_mean_torch = torch.mean(normal_vectors, dim=0)
                                normal_vectors_cov_torch = torch.cov(normal_vectors.transpose(0, 1))

                                def mahal(u, v, cov):
                                    delta = u - v
                                    m = torch.dot(delta, torch.matmul(cov, delta))
                                    return torch.sqrt(m)

                                mahalanobis_dists = [mahal(feat, normal_vector_mean_torch, normal_vectors_cov_torch) for
                                                     feat in normal_vectors]
                                dist_max = torch.tensor(mahalanobis_dists).max()
                                dist_mean = torch.tensor(mahalanobis_dists).mean()
                                dist_loss += dist_mean

                                random_anomal_positions = []
                                for i in range(pix_num) :
                                    object_flag = object_position[i]
                                    feat = random_vectors[i, :]
                                    random_dist = mahal(feat, normal_vector_mean_torch, normal_vectors_cov_torch)
                                    if object_flag == 1 :
                                        if random_dist >= dist_max:
                                            random_anomal_positions.append(1)
                                        else:
                                            random_anomal_positions.append(0)
                                    else :
                                        random_anomal_positions.append(0)
                                random_anomal_positions = torch.tensor(random_anomal_positions)
                                random_anomal_positions = random_anomal_positions.unsqueeze(0).repeat(head_num,1).\
                                    to(score_random_map.device)

                                normal_trigger_activation = (score_map * normal_position)
                                normal_trigger_back_activation = (score_map * back_position)
                                anormal_trigger_activation = (score_random_map * random_anomal_positions)
                                total_score = torch.ones_like(anormal_trigger_activation)

                                normal_trigger_activation = normal_trigger_activation.sum(dim=-1)  # 8
                                normal_trigger_back_activation = normal_trigger_back_activation.sum(dim=-1)  # 8
                                anormal_trigger_activation = anormal_trigger_activation.sum(dim=-1)  # 8
                                total_score = total_score.sum(dim=-1)  # 8

                                normal_trigger_activation_loss = (1 - (normal_trigger_activation / total_score)) ** 2  # 8, res*res
                                normal_trigger_back_activation_loss = (normal_trigger_back_activation / total_score) ** 2  # 8, res*res
                                anormal_trigger_activation_loss = (anormal_trigger_activation / total_score) ** 2  # 8, res*res

                                activation_loss = args.normal_weight * normal_trigger_activation_loss\
                                                  + args.back_weight * normal_trigger_back_activation_loss
                                # ---------------------------------- deactivating ------------------------------------ #
                                if args.act_deact :
                                    activation_loss += args.act_deact_weight * anormal_trigger_activation_loss
                                if args.cls_training:
                                    normal_cls_activation = (cls_map * normal_position).sum(dim=-1)
                                    normal_cls_back_activation = (cls_map * back_position).sum(dim=-1)
                                    anormal_cls_activation = (cls_random_map * random_anomal_positions).sum(dim=-1)

                                    normal_cls_activation_loss = (normal_cls_activation / total_score) ** 2
                                    normal_cls_back_activation_loss = (1 - (normal_cls_back_activation / total_score)) ** 2
                                    anormal_cls_activation_loss = (1 - (anormal_cls_activation / total_score)) ** 2
                                    activation_loss += args.normal_weight * normal_cls_activation_loss \
                                                     + args.back_weight * normal_cls_back_activation_loss
                                    if args.act_deact :
                                        activation_loss += args.act_deact_weight * anormal_cls_activation_loss
                                attn_loss += activation_loss
                dist_loss = dist_loss.mean()
                attn_loss = attn_loss.mean()
                ########################### 3. attn loss ###########################################################
                loss = args.task_loss_weight * task_loss + args.mahalanobis_loss_weight * dist_loss + args.attn_loss_weight * attn_loss
                # ------------------------------------------------------------------------------------------------- #
                if is_main_process:
                    loss_dict["loss/task_loss"] = task_loss.item()
                    loss_dict["loss/attn_loss"] = attn_loss.item()
                    loss_dict["loss/dist_loss_1"] = dist_loss.item()
                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = network.get_trainable_params()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # ------------------------------------------------------------------------------------------------- #

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    #self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer,
                    #                   text_encoder, unet)
                    attention_storer.reset()

                    # 指定ステップごとにモデルを保存
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
                # ------------------------------------------------------------------------------------------------------
                # 1) total loss
                current_loss = loss.detach().item()
                if epoch == args.start_epoch:
                    loss_list.append(current_loss)
                else:
                    loss_total -= loss_list[step]
                    loss_list[step] = current_loss
                loss_total += current_loss
                avr_loss = loss_total / len(loss_list)
                if is_main_process:
                    progress_bar.set_postfix(**loss_dict)
                if global_step >= args.max_train_steps:
                    break
            accelerator.wait_for_everyone()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (
                        epoch + 1) < args.start_epoch + num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)
                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as,
                                                                          remove_epoch_no)
                        remove_model(remove_ckpt_name)
                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)
            #self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer,
            #                   text_encoder, unet)
            attention_storer.reset()
        if is_main_process:
            network = accelerator.unwrap_model(network)
        accelerator.end_training()
        if is_main_process and args.save_state:
            train_util.save_state_on_train_end(args, accelerator)


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
    parser.add_argument("--resume_lora_training", action="store_true", )
    parser.add_argument("--back_training", action="store_true", )
    parser.add_argument("--back_weight", type=float, default=1)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--valid_data_dir", type=str)
    parser.add_argument("--task_loss_weight", type=float, default=0.5)
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--truncate_pad", action='store_true')
    parser.add_argument("--truncate_length", type=int, default=3)
    parser.add_argument("--detail_64_up", action='store_true')
    parser.add_argument("--detail_64_down", action='store_true')
    parser.add_argument("--anormal_sample_normal_loss", action='store_true')
    parser.add_argument("--do_task_loss", action='store_true')
    parser.add_argument("--act_deact", action='store_true')
    parser.add_argument("--act_deact_weight", type=float, default=1.0)
    parser.add_argument("--concat_query", action='store_true')
    parser.add_argument("--strict_training", action='store_true')
    parser.add_argument("--partial", action="store_true", )
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--trg_part", type=arg_as_list, default=['down', 'up'])
    parser.add_argument("--trg_layer", type=str)
    parser.add_argument("--trg_layer_list", type=arg_as_list, )
    # default=['down_blocks_0_attentions_1_transformer_blocks_0_attn2',
    #         'up_blocks_3_attentions_2_transformer_blocks_0_attn2'])
    parser.add_argument("--training_layer", type=str)
    parser.add_argument('--trg_position', type=arg_as_list, default=['down', 'up'])
    parser.add_argument('--mahalanobis_loss_weight', type=float, default=1.0)
    parser.add_argument("--attn_loss_weight", type=float, default=1.0)
    parser.add_argument('--normal_weight', type=float, default=1.0)
    parser.add_argument("--cross_map_res", type=arg_as_list, default=[64, 32, 16, 8])
    parser.add_argument("--cls_training", action="store_true", )
    parser.add_argument("--background_loss", action="store_true")
    parser.add_argument("--average_mask", action="store_true", )
    parser.add_argument("--attn_loss", action="store_true", )
    parser.add_argument("--normal_with_background", action="store_true", )
    parser.add_argument("--only_object_position", action="store_true", )
    parser.add_argument("--query_add_random", action="store_true", )
    parser.add_argument("--unet_frozen", action="store_true", )
    parser.add_argument("--text_frozen", action="store_true", )
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    trainer = NetworkTrainer()
    trainer.train(args)