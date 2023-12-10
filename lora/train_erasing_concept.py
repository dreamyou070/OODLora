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
from library import model_util
import library.train_util as train_util
from library.train_util import (DreamBoothDataset, )
import library.config_util as config_util
from library.config_util import (ConfigSanitizer, BlueprintGenerator, )
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (apply_snr_weight, get_weighted_text_embeddings, prepare_scheduler_for_custom_training)
from setproctitle import *
from PIL import Image
import numpy as np

global_stored_masks = {}


def get_cached_mask(mask_dir: str, trg_size):
    # if mask_dir in global_stored_masks:
    #    return global_stored_masks[mask_dir]
    pil_img = Image.open(mask_dir)
    pil_img = pil_img.resize((trg_size, trg_size))
    np_img = np.array(pil_img)
    torch_img = torch.from_numpy(np_img)
    mask_img = torch.where(torch_img == 0, 0, 1)
    global_stored_masks[mask_dir] = mask_img
    return mask_img

def arg_as_list(s):
    import ast
    v = ast.literal_eval(s)
    return v


class NetworkTrainer:

    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(self,
                           args: argparse.Namespace, current_loss, avr_loss,
                           lr_scheduler, keys_scaled=None, mean_norm=None, maximum_norm=None,
                           task_loss=None, attn_loss=None, ):
        if task_loss and attn_loss:
            logs = {"loss/task_loss": task_loss.item(),
                    "loss/attn_loss": attn_loss.item(),
                    "loss/current": current_loss,
                    "loss/average": avr_loss, }
        else:
            logs = {"loss/task_loss": task_loss.item(),
                    "loss/current": current_loss,
                    "loss/average": avr_loss, }

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
                    "lr"]
                )
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
                            lr_scheduler.optimizers[-1].param_groups[i]["lr"]
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

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, tokenizers, text_encoders, data_loader,
                                             weight_dtype):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizers[0], text_encoders[0],
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

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet,
                      attention_storer):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet,
                                 attention_storer=attention_storer)

    def train(self, args):

        print(f'\n step 1. setting')
        print(f' (1.1) process title')
        if args.process_title:
            setproctitle(args.process_title)
        else:
            setproctitle('parksooyeon')
        session_id = random.randint(0, 2 ** 32)
        print(f' (1.2) seed')
        if args.seed is None: args.seed = random.randint(0, 2 ** 32)
        set_seed(args.seed)
        print(f' (1.3) accelerator')
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process
        print(f' (1.4) save directory and save config')
        save_base_dir = args.output_dir
        _, folder_name = os.path.split(save_base_dir)
        record_save_dir = os.path.join(args.output_dir, "record")
        os.makedirs(record_save_dir, exist_ok=True)
        with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f' (1.5) make wandb dir')
        wandb.login(key=args.wandb_api_key)
        if is_main_process:
            wandb.init(project=args.wandb_init_name)
            wandb.run.name = folder_name

        # --------------------------------------------------------------------------------------------------------------------------------------
        print(f'\n step 2. make model')
        print(f' (2.1) mixed precision and model')
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
        print(f' (2.2) loading model')
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        print(f' (2.3) get lora network')
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)
        if args.base_weights is not None:
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]
                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")
                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True)
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype,
                                accelerator.device if args.lowram else "cpu")
            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, weights_sd = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet,**net_kwargs)
        else:
            network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae,text_encoder, unet, neuron_dropout=args.network_dropout,**net_kwargs, )
        if network is None:
            return
        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            print("warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません")
            args.scale_weight_norms = False
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)
        # 学習に必要なクラスを準備する
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
        accelerator.print("prepare optimizer, data loader etc.")
        from gen_img_diffusers import PipelineLike
        device = accelerator.device if args.lowram else "cpu"

        print(f'\n step 3. dataset')
        train_util.prepare_dataset_args(args, True)
        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None
        use_class_caption = args.class_caption is not None
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
            if use_user_config:
                print(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    print(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                            ", ".join(ignored)))
            else:
                if use_dreambooth_method:
                    print("Using DreamBooth method.")
                    user_config = {}
                    user_config['datasets'] = [{"subsets": None}]
                    subsets_dict_list = []
                    for subsets_dict in config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir,
                                                                                                  args.reg_data_dir):
                        if use_class_caption:
                            subsets_dict['class_caption'] = args.class_caption
                        subsets_dict_list.append(subsets_dict)
                        user_config['datasets'][0]['subsets'] = subsets_dict_list
                else:
                    print("Training with captions.")
                    user_config = {}
                    user_config["datasets"] = []
                    user_config["datasets"].append({"subsets": [{"image_dir": args.train_data_dir,
                                                                 "metadata_file": args.in_json, }]})
                    if use_class_caption:
                        for subset in user_config["datasets"][0]["subsets"]:
                            subset["class_caption"] = args.class_caption
            print(f'User config: {user_config}')
            # blueprint_generator = BlueprintGenerator
            print('start of generate function ...')
            blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

        training_started_at = time.time()
        train_util.verify_training_args(args)
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)
        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            print("No data found. Please verify arguments (train_data_dir must be the parent of folders with images) ）")
            return
        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
        self.assert_extra_args(args, train_dataset_group)

        """
        try:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        except TypeError:
            accelerator.print(
                "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)")
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)
        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)
        """
        # --------------------------------------------------------------------------------------------------------------------------------------





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    train_util.add_training_arguments(parser, True)
    train_util.add_sd_models_arguments(parser)

    parser.add_argument('--process_title', type=str, default='parksooyeon')
    # step 2. make model
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
    parser.add_argument("--network_train_unet_only", action="store_true",
                       help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
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


    # step 5. optimizer
    train_util.add_optimizer_arguments(parser)
    # step 3. dataset common
    train_util.add_dataset_arguments(parser, True, True, True)
    # step 4. training
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")
    args = parser.parse_args()
    trainer = NetworkTrainer()
    trainer.train(args)