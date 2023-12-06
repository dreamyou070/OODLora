from collections import defaultdict
import importlib, wandb
import argparse
import gc
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
import toml
from torch import nn
from tqdm import tqdm
import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from lora.library import model_util
from lora import library as train_util, library as config_util, library as huggingface_util, \
    library as custom_train_functions
from lora.library import (DreamBoothDataset, )
from lora.library import (ConfigSanitizer, BlueprintGenerator, )
from lora.library import (apply_snr_weight, get_weighted_text_embeddings, prepare_scheduler_for_custom_training,
                          scale_v_prediction_loss_like_noise_prediction, add_v_prediction_like_loss, )
from setproctitle import *
from utils import _convert_heat_map_colors
from PIL import Image
import numpy as np
import torch.nn.functional as F
from utils import auto_autocast

global_stored_masks = {}
def get_cached_mask(mask_dir:str, trg_size):
    #if mask_dir in global_stored_masks:
    #    return global_stored_masks[mask_dir]
    pil_img = Image.open(mask_dir)
    pil_img = pil_img.resize((trg_size,trg_size))
    np_img = np.array(pil_img)
    torch_img = torch.from_numpy(np_img)
    mask_img = torch.where(torch_img== 0, 0, 1)
    global_stored_masks[mask_dir] = mask_img
    return mask_img

factors = {0, 1, 2, 4, 8, 16, 32, 64}

def register_attention_control(unet : nn.Module, controller):
    """
    Register cross attention layers to controller.
    """
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
            """
            if is_cross_attention:
                if trg_indexs_list is not None:
                    trg_indexs = trg_indexs_list
                    batch_num = len(trg_indexs)
                    attention_probs_batch = torch.chunk(attention_probs, batch_num, dim=0)
                    for batch_idx, attention_prob in enumerate(attention_probs_batch) :
                        batch_trg_index = trg_indexs[batch_idx] # two times
                        head_num = attention_prob.shape[0]
                        res = int(math.sqrt(attention_prob.shape[1]))
                        for word_idx in batch_trg_index :
                            # head, pix_len
                            word_heat_map = attention_prob[:, :, word_idx]
                            word_heat_map_ = word_heat_map.reshape(-1, res, res)
                            word_heat_map_ = word_heat_map_.mean(dim=0)
                            # word_heat_map = (512,512)
                            word_heat_map_ = F.interpolate(word_heat_map_.unsqueeze(0).unsqueeze(0),size=((512, 512)),mode='bicubic').squeeze()
                            mask_ = mask[batch_idx].to(attention_prob.dtype) # (512,512)
                            masked_heat_map = word_heat_map_ * mask_
                            #heat_map = _convert_heat_map_colors(masked_heat_map)
                            #heat_map = heat_map.to('cpu').detach().numpy().copy().astype(np.uint8)
                            #heat_map_img = Image.fromarray(heat_map)
                            #heat_map_img.save(os.path.join(args.output_dir, f'heatmap_{layer_name}.png'))
                            #torch.save(masked_heat_map, os.path.join(args.output_dir, f'heatmap_{layer_name}.pth'))
                            attn_loss = F.mse_loss(word_heat_map_.mean(), masked_heat_map.mean())
                            controller.store(attn_loss, layer_name)
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

def arg_as_list(s):
    import ast
    v = ast.literal_eval(s)
    return v


def generate_text_embedding(prompt, tokenizer, text_encoder, batch):
    cls_token = 49406
    pad_token = 49407
    trg_token = args.trg_token
    token_input = tokenizer([trg_token],
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt", )
    token_ids = token_input.input_ids[0]
    token_attns = token_input.attention_mask[0]
    trg_token_id = []
    for token_id, token_attn in zip(token_ids, token_attns):
        if token_id != cls_token and token_id != pad_token and token_attn == 1:
            trg_token_id.append(token_id)
    text_input = tokenizer(batch['captions'],
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt", )
    token_ids = text_input.input_ids
    attns = text_input.attention_mask
    batch_ids = []
    for token_id, attn in zip(token_ids, attns):
        trg_indexs = []
        for i, id in enumerate(token_id):
            if id in trg_token_id:
                trg_indexs.append(i)
        batch_ids.append(trg_indexs)
    return batch_ids

class NetworkTrainer:

    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(self,
                           args: argparse.Namespace, current_loss, avr_loss,
                           lr_scheduler, keys_scaled=None, mean_norm=None, maximum_norm=None,
                           task_loss=None, attn_loss=None,):
        if task_loss and attn_loss :
            logs = {"loss/task_loss": task_loss.item(),
                    "loss/attn_loss": attn_loss.item(),
                    "loss/current": current_loss,
                    "loss/average": avr_loss,}
        else :
            logs = {"loss/task_loss": task_loss.item(),
                    "loss/current": current_loss,
                    "loss/average": avr_loss,}

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
                args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                    lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                )
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )

        return logs

    def assert_extra_args(self, args, train_dataset_group):
        pass

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_tokenizer(self, args):
        tokenizer = train_util.load_tokenizer(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return False

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, tokenizers, text_encoders, data_loader, weight_dtype):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizers[0], text_encoders[0], weight_dtype)
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
                          mask_imgs=mask_imgs,).sample
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet,attention_storer):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet, attention_storer=attention_storer)

    def train(self, args):

        if args.process_title :
            setproctitle(args.process_title)
        else :
            setproctitle('parksooyeon')

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # tokenizerは単体またはリスト、tokenizersは必ずリスト：既存のコードとの互換性のため
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
            if use_user_config:
                print(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    print("ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(", ".join(ignored)))
            else:
                if use_dreambooth_method:
                    print("Using DreamBooth method.")
                    user_config = {"datasets": [{"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)}]}
                else:
                    print("Training with captions.")
                    user_config = {"datasets": [{"subsets": [{"image_dir": args.train_data_dir,"metadata_file": args.in_json,}]}]}
            blueprint = blueprint_generator.generate(user_config,
                                                     args,
                                                     tokenizer=tokenizer)
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)
        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            print(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        self.assert_extra_args(args, train_dataset_group)

        # acceleratorを準備する
        print("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        save_base_dir = args.output_dir
        _, folder_name = os.path.split(save_base_dir)
        # save config
        record_save_dir = os.path.join(args.output_dir, "record")
        os.makedirs(record_save_dir, exist_ok=True)
        with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        wandb.login(key=args.wandb_key)
        if is_main_process:
            print(" make wandb process log file")
            wandb.init(project=args.wandb_init_name)
            wandb.run.name = folder_name
            """
            cfg = {"attn_loss_ratio": 1}
            sweep_id = wandb.sweep(cfg.sweep_config)
            wandb.config.update(cfg)
            sweep_configuration = {"method": "random",
                                   "metric": {"goal": "minimize", "name": "loss"},
                                   "parameters": {"attnloss_ratio": {"max": 20, "min": 1},},}
            sweep_id = wandb.sweep(sweep=sweep_configuration,
                                   project="sy-first-sweep")
            wandb.agent(sweep_id, function=, count=10)
            """


        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
        # モデルを読み込む
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)
        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)
        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]
                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")
                module, weights_sd = network_module.create_network_from_weights(multiplier, weight_path,
                                                                                args.block_wise,
                                                                                vae,
                                                                                text_encoder, unet, for_inference=True)
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")
            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # 学習を準備する
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad():
                train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
            vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            accelerator.wait_for_everyone()
        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, tokenizers, text_encoders,
                                                  train_dataset_group, weight_dtype)

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value
        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights,
                                                                    args.block_wise,
                                                                    vae, text_encoder, unet,
                                                                    **net_kwargs)

        else:
            print(f'trg block_wise: {args.block_wise}')
            network = network_module.create_network_blockwise(
                    1.0,
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    text_encoder,
                    unet,
                    block_wise=args.block_wise,
                    neuron_dropout=args.network_dropout,
                    **net_kwargs,)
        if network is None:
            return
        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            print(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for t_enc in text_encoders:
                t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        if args.unet_blockwise_lr :
            network.set_block_lr_weight(up_lr_weight   = args.up_lr_weight, # 0 ~ 11
                                        mid_lr_weight  = args.mid_lr_weight,
                                        down_lr_weight = args.down_lr_weight)
        try:
            trainable_params = network.prepare_optimizer_params(text_encoder_lr=args.text_encoder_lr,
                                                                unet_lr=args.unet_lr,
                                                                default_lr=args.learning_rate)
        except TypeError:
            accelerator.print("Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)")
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで
        train_dataloader = torch.utils.data.DataLoader(train_dataset_group,batch_size=1,
                                                       shuffle=True,collate_fn=collater,num_workers=n_workers,
                                                       persistent_workers=args.persistent_data_loader_workers,)
        # 学習ステップ数を計算する
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")
        # データセット側にも学習ステップを送信
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr schedulerを用意する
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet.requires_grad_(False)
        unet.to(dtype=weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)
        
        # acceleratorがなんかよろしくやってくれるらしい
        # TODO めちゃくちゃ冗長なのでコードを整理する
        if train_unet and train_text_encoder:
            if len(text_encoders) > 1:
                unet, t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    unet, text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler
                )
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
            else:
                unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler
                )
                text_encoders = [text_encoder]
        elif train_unet:
            unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, network, optimizer, train_dataloader, lr_scheduler
            )
        elif train_text_encoder:
            if len(text_encoders) > 1:
                t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler
                )
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
            else:
                text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    text_encoder, network, optimizer, train_dataloader, lr_scheduler
                )
                text_encoders = [text_encoder]

            unet.to(accelerator.device, dtype=weight_dtype)  # move to device because unet is not prepared by accelerator
        else:
            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, lr_scheduler
            )

        # transform DDP after prepare (train_network here only)
        text_encoders = train_util.transform_models_if_DDP(text_encoders)
        unet, network = train_util.transform_models_if_DDP([unet, network])

        if args.gradient_checkpointing:
            unet.train()
            for t_enc in text_encoders:
                t_enc.train()
                t_enc.text_model.embeddings.requires_grad_(True)
        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()
        del t_enc
        network.prepare_grad_etc(text_encoder, unet)
        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets.

        from attention_store import AttentionStore
        attention_storer = AttentionStore()
        register_attention_control(unet, attention_storer)


        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
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
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
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
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
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

        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

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
                    "ss_bucket_info": json.dumps(dataset.bucket_info),})

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
            accelerator.init_trackers("network_train" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs)
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
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # training loop
        if is_main_process :
            gradient_dict = {}
            loss_dict = {}

        with open(os.path.join(args.output_dir, "attnloss.txt"), "a") as f:
            for epoch in range(num_train_epochs):
                accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
                current_epoch.value = epoch + 1
                metadata["ss_epoch"] = str(epoch + 1)
                network.on_epoch_start(text_encoder, unet)
                for step, batch in enumerate(train_dataloader):
                    current_step.value = global_step
                    with accelerator.accumulate(network):
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
                        b_size = latents.shape[0]
                        with torch.set_grad_enabled(train_text_encoder):
                            if args.weighted_captions:
                                text_encoder_conds = get_weighted_text_embeddings(tokenizer,text_encoder, batch["captions"],accelerator.device,
                                                                                  args.max_token_length // 75 if args.max_token_length else 1,
                                                                                  clip_skip=args.clip_skip,)
                            else:
                                text_encoder_conds = self.get_text_cond(args,
                                                                        accelerator,
                                                                        batch,
                                                                        tokenizers,
                                                                        text_encoders,
                                                                        weight_dtype)
                        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args,noise_scheduler,latents)
                        # Predict the noise residual
                        with accelerator.autocast():
                            # -----------------------------------------------------------------------------------------------------------------------
                            noise_pred = self.call_unet(args,
                                                        accelerator,
                                                        unet,
                                                        noisy_latents,
                                                        timesteps,
                                                        text_encoder_conds,
                                                        batch,
                                                        weight_dtype,
                                                        batch["trg_indexs_list"],
                                                        batch['mask_imgs'])
                            # -----------------------------------------------------------------------------------------------------------------------
                            atten_collection = attention_storer.step_store
                            attention_storer.reset()
                            attention_storer.step_store = {}

                            self_query_collection = attention_storer.self_query_store
                            self_key_collection = attention_storer.self_key_store
                            cross_key_collection = attention_storer.cross_key_store
                            attention_storer.self_query_store = {}
                            attention_storer.self_key_store = {}
                            attention_storer.cross_key_store = {}

                        if args.v_parameterization:
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            target = noise
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                        loss = loss.mean([1, 2, 3])
                        loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                        loss = loss * loss_weights
                        if args.min_snr_gamma:
                            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                        if args.scale_v_pred_loss_like_noise_pred:
                            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                        if args.v_pred_like_loss:
                            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                        loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし
                        task_loss = loss

                        # ------------------------------------------------------------------------------------
                        if args.heatmap_loss :
                            layer_names = atten_collection.keys()
                            attn_loss = 0
                            for layer_name in layer_names:
                                if 'down_blocks_2' not in layer_name or 'mid' not in layer_name or 'up_blocks_1' not in layer_name :
                                    a = sum(atten_collection[layer_name])
                                    attn_loss = attn_loss + sum(atten_collection[layer_name])
                                    #print(f'{layer_name} : {a.item()}')
                                    f.write(f'{layer_name} : {a.item()}\n')
                            loss = task_loss + args.attn_loss_ratio * attn_loss
                            #loss = args.attn_loss_ratio * attn_loss
                        accelerator.backward(loss)
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            params_to_clip = network.get_trainable_params()
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm )
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    if args.scale_weight_norms:
                        keys_scaled, mean_norm, maximum_norm = network.apply_max_norm_regularization(args.scale_weight_norms,
                                                                                                     accelerator.device)
                        max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                    else:
                        keys_scaled, mean_norm, maximum_norm = None, None, None

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet, attention_storer=attention_storer)
                        attention_storer.reset()
                        attention_storer.step_store = {}
                        attention_storer.self_query_store = {}
                        attention_storer.self_key_store = {}
                        attention_storer.cross_key_store = {}

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
                                    remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                    remove_model(remove_ckpt_name)
                    # ------------------------------------------------------------------------------------------------------
                    # 1) total loss
                    current_loss = loss.detach().item()
                    if epoch == 0:
                        loss_list.append(current_loss)
                    else:
                        loss_total -= loss_list[step]
                        loss_list[step] = current_loss
                    loss_total += current_loss
                    avr_loss = loss_total / len(loss_list)
                    if is_main_process :
                        loss_dict[global_step] = avr_loss
                    logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    if args.scale_weight_norms:
                        progress_bar.set_postfix(**{**max_mean_logs, **logs})
                    # ------------------------------------------------------------------------------------------------------
                    # 2) total loss
                    if args.logging_dir is not None:
                        if args.heatmap_loss :
                            logs = self.generate_step_logs(args, current_loss, avr_loss,
                                                           lr_scheduler, keys_scaled, mean_norm, maximum_norm,
                                                           task_loss, attn_loss)
                        else :
                            logs = self.generate_step_logs(args, current_loss, avr_loss,
                                                           lr_scheduler, keys_scaled, mean_norm, maximum_norm,task_loss)
                        accelerator.log(logs, step=global_step)
                        if is_main_process:
                            wandb.log(logs, step=global_step)
                    if global_step >= args.max_train_steps:
                        break
                if args.logging_dir is not None:
                    if args.heatmap_loss:
                        logs = {"loss/epoch": loss_total / len(loss_list),
                                "loss/task_loss": task_loss.item(),
                                "loss/attn_loss": attn_loss.item()}
                    else:
                        logs = {"loss/epoch": loss_total / len(loss_list),}
                    accelerator.log(logs, step=epoch + 1)
                accelerator.wait_for_everyone()
                # 指定エポックごとにモデルを保存
                if args.save_every_n_epochs is not None:
                    saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                    if is_main_process and saving:
                        ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                        if args.unwrap :
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)
                        else :
                            save_model(ckpt_name, network, global_step, epoch + 1)
                        remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                        if remove_epoch_no is not None:
                            remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                            remove_model(remove_ckpt_name)
                        if args.save_state:
                            train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

                self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet, attention_storer=attention_storer)
                attention_storer.reset()
                attention_storer.step_store = {}
                attention_storer.self_query_store = {}
                attention_storer.self_key_store = {}
                attention_storer.cross_key_store = {}
        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)
        accelerator.end_training()
        if is_main_process and args.save_state:
            train_util.save_state_on_train_end(args, accelerator)
        if is_main_process:
            print("model saved.")
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            print("gradient recording")
            gradient_save_dir = os.path.join(record_save_dir, "gradient_norm.pickle")
            import pickle
            with open(gradient_save_dir, 'wb') as fw:
                pickle.dump(gradient_dict, fw)

            print("loss recording")
            loss_save_dir = os.path.join(record_save_dir, "loss.pickle")
            with open(loss_save_dir, 'wb') as fw:
                pickle.dump(loss_dict, fw)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument("--save_model_as",type=str,default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",)
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--network_dim", type=int, default=None, help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）")
    parser.add_argument("--network_alpha",type=float,default=1,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",)
    parser.add_argument("--network_dropout",type=float,default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",)
    parser.add_argument("--network_args", type=str, default=None, nargs="*", help="additional argmuments for network (key=value) / ネットワークへの追加の引数")
    parser.add_argument("--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument(
        "--network_train_text_encoder_only", action="store_true", help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument(
        "--training_comment", type=str, default=None, help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
    parser.add_argument("--dim_from_weights",action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",)
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    parser.add_argument("--block_wise", type = arg_as_list,
                        default = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,])
    parser.add_argument("--unwrap", action = 'store_true')
    parser.add_argument("--process_title", type=str, default = 'parksooyeon')
    parser.add_argument("--wandb_init_name", type=str)
    parser.add_argument("--wandb_key", type=str)
    parser.add_argument("--unet_blockwise_lr", action = 'store_true')
    parser.add_argument("--trg_token", type=str, default = 'haibara')
    parser.add_argument("--heatmap_loss", action = 'store_true')
    parser.add_argument("--attn_loss_ratio", type = float, default = 1.0)
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    trainer = NetworkTrainer()
    trainer.train(args)