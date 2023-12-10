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

        net_key_names = args.net_key_names
        net_kwargs['key_layers'] = net_key_names.split(",")
        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet,
                                                                    **net_kwargs)
        else:
            # LyCORIS will work with this...
            network = network_module.create_network(1.0,
                                                    args.network_dim, args.network_alpha, vae,
                                                    text_encoder, unet, neuron_dropout=args.network_dropout,
                                                    **net_kwargs, )
        if network is None:
            return
        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            print(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません")
            args.scale_weight_norms = False
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)

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
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)
        accelerator.print("prepare optimizer, data loader etc.")
        # 後方互換性を確保するよ
        try:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        except TypeError:
            accelerator.print(
                "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)")
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)
        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        # --------------------------------------------------------------------------------------------------------------------------------------
        print(f'\n step 3. dataset')
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        user_config = {"datasets": [
            {"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir,
                                                                                  args.reg_data_dir)}]}
        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        if args.cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad():
                train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk,
                                                  accelerator.is_main_process)
            vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            accelerator.wait_for_everyone()

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, tokenizers, text_encoders,
                                                  train_dataset_group, weight_dtype)









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    parser.add_argument('--process_title', type=str, default='parksooyeon')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--logging_dir", type=str, default=None, )
    parser.add_argument("--log_with", type=str, default='wandb', choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None,
                        help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列")
    parser.add_argument('--wandb_api_key', type=str, default='3a3bc2f629692fa154b9274a5bbe5881d47245dc')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数", )
    parser.add_argument('--output_dir', type=str, default='wandb')
    parser.add_argument('--wandb_init_name', type=str, default='wandb')
    # step 2. make model
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="use mixed precision / 混合精度を使う場合、その精度")
    parser.add_argument("--save_precision",type=str,default=None,choices=[None, "float", "fp16", "bf16"],
                        help="precision in saving / 保存時に精度を変更して保存する",)
    parser.add_argument("--no_half_vae",)
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
    # step 3. dataset common
    parser.add_argument("--train_data_dir", type=str, default=None,
                        help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--shuffle_caption", action="store_true",
                        help="shuffle comma-separated caption / コンマで区切られたcaptionの各要素をshuffleする")
    parser.add_argument("--caption_extension", type=str, default=".caption",
                        help="extension of caption files / 読み込むcaptionファイルの拡張子")
    parser.add_argument("--caption_extention", type=str, default=None,
                        help="extension of caption files (backward compatibility) / 読み込むcaptionファイルの拡張子（スペルミスを残してあります）", )
    parser.add_argument("--keep_tokens", type=int, default=0,
                        help="keep heading N tokens when shuffling caption tokens (token means comma separated strings) / captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す（トークンはカンマ区切りの各部分を意味する）", )
    parser.add_argument("--caption_prefix", type=str, default=None,
                        help="prefix for caption text / captionのテキストの先頭に付ける文字列", )
    parser.add_argument("--caption_suffix", type=str, default=None,
                        help="suffix for caption text / captionのテキストの末尾に付ける文字列", )
    parser.add_argument("--color_aug", action="store_true",
                        help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする")
    parser.add_argument("--flip_aug", action="store_true",
                        help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする")
    parser.add_argument("--face_crop_aug_range", type=str, default=None,
                        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0) / 学習時に顔を中心とした切り出しaugmentationを有効にするときは倍率を指定する（例：2.0,4.0）", )
    parser.add_argument("--random_crop", action="store_true",
                        help="enable random crop (for style training in face-centered crop augmentation) / ランダムな切り出しを有効にする（顔を中心としたaugmentationを行うときに画風の学習用に指定する）", )
    parser.add_argument("--debug_dataset", action="store_true",
                        help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）")
    parser.add_argument("--resolution", type=str, default=None,
                        help="resolution in training ('size' or 'width,height') / 学習時の画像解像度（'サイズ'指定、または'幅,高さ'指定）", )
    parser.add_argument("--cache_latents", action="store_true",
                        help="cache latents to main memory to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをメインメモリにcacheする（augmentationは使用不可） ", )
    parser.add_argument("--vae_batch_size", type=int, default=1,
                        help="batch size for caching latents / latentのcache時のバッチサイズ")
    parser.add_argument("--cache_latents_to_disk", action="store_true",
                        help="cache latents to disk to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをディスクにcacheする（augmentationは使用不可）", )
    parser.add_argument("--enable_bucket", action="store_true",
                        help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする")
    parser.add_argument("--min_bucket_reso", type=int, default=256,
                        help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024,
                        help="maximum resolution for buckets / bucketの最大解像度")
    parser.add_argument("--bucket_reso_steps", type=int, default=64,
                        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します", )
    parser.add_argument("--bucket_no_upscale", action="store_true",
                        help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します")
    parser.add_argument("--token_warmup_min", type=int, default=1,
                        help="start learning at N tags (token means comma separated strinfloatgs) / タグ数をN個から増やしながら学習する", )
    parser.add_argument("--token_warmup_step", type=float, default=0,
                        help="tag length reaches maximum on N steps (or N*max_train_steps if N<1) / N（N<1ならN*max_train_steps）ステップでタグ長が最大になる。デフォルトは0（最初から最大）", )
    parser.add_argument("--dataset_class",type=str,default=None,
                        help="dataset class for arbitrary dataset (package.module.Class) / 任意のデータセットを用いるときのクラス名 (package.module.Class)",)
    support_caption_dropout = True
    if support_caption_dropout:
        parser.add_argument("--caption_dropout_rate", type=float, default=0.0,
                            help="Rate out dropout caption(0.0~1.0) / captionをdropoutする割合")
        parser.add_argument("--caption_dropout_every_n_epochs",
                            type=int,default=0,help="Dropout all captions every N epochs / captionを指定エポックごとにdropoutする",)
        parser.add_argument("--caption_tag_dropout_rate",type=float,default=0.0,
                            help="Rate out dropout comma separated tokens(0.0~1.0) / カンマ区切りのタグをdropoutする割合",)
    support_dreambooth = True
    if support_dreambooth:
        parser.add_argument("--reg_data_dir", type=str, default=None,
                            help="directory for regularization images / 正則化画像データのディレクトリ")
    support_caption = True
    if support_caption:
        parser.add_argument("--in_json", type=str, default=None,
                            help="json metadata for dataset / データセットのmetadataのjsonファイル")
        parser.add_argument("--dataset_repeats", type=int, default=1,
                            help="repeat dataset when training with captions / キャプションでの学習時にデータセットを繰り返す回数")
    args = parser.parse_args()
    trainer = NetworkTrainer()
    trainer.train(args)