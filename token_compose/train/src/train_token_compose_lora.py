import argparse
import logging
import math
import os
import random
import shutil
import random
import itertools
import wandb
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import diffusers
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from attn_utils import AttentionStore
from attn_utils import register_attention_control, get_cross_attn_map_from_unet
from loss_utils import get_grounding_loss_by_layer, get_word_idx
from data_utils import DatasetPreprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from networks.lora import create_network
from utils import train_util

logger = get_logger(__name__, log_level="INFO")


def main(args):

    print(f'\n step 1. checking train layer ls')
    train_layers_ls = [f"down_{res}" for res in args.train_down] + \
                      [f"mid_{res}" for res in args.train_mid] + [f"up_{res}" for res in args.train_up]

    print(f' - train_layers_ls: {train_layers_ls}')

    print(f'\n step 2. make accelerate')
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    mixed_precision = None
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=mixed_precision, log_with=args.report_to,
                              project_config=accelerator_project_config, )

    print(f'\n step 3. log on every process with the configuration for debugging.')
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, )
    logger.info(accelerator.state, main_process_only=False)

    print(f'\n step 4. resume or not')
    # change output dir first
    if args.resume_from_checkpoint:
        # change the output dir manually
        resume_ckpt_number = args.resume_from_checkpoint.split("-")[-1]
        args.output_dir = f"{args.output_dir}-resume-{resume_ckpt_number}"
        logger.info(f"change output dir to {args.output_dir}")

    print(f'\n step 5. set logging level (only main warning, other error)')
    # level : ALL < DEBUG < INFO < WARN < ERROR < FATAL < OFF (logging just higher level)
    # log level warning : warn, error, fatal
    # log level info : info, warn, error, fatal
    # log level error : error, fatal
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    print(f'\n step 6. make output dir')
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    print(f'\n step 7. models and lora model')
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    net_kwargs = {}
    network = create_network(1.0, args.network_dim,
                             args.network_alpha,
                             vae,
                             text_encoder,
                             unet,
                             neuron_dropout=args.network_dropout, **net_kwargs, )
    print(' (7.1) lora with unet and text encoder')
    train_unet = not args.network_train_text_encoder_only
    train_text_encoder = not args.network_train_unet_only
    network.apply_to(text_encoder, unet, train_text_encoder, train_unet)
    print(' (7.2) lora resume?')
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        accelerator.print(f"load network weights from {args.network_weights}: {info}")
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
        network.enable_gradient_checkpointing()  # may have no effect

    print(f'\n step 8. optimizer')
    try:
        trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    except:
        trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)
    optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)




if __name__ == "__main__":
    # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="CompVis/stable-diffusion-v1-4",
                        #choices=["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"],
                        help="Path to pretrained model or model identifier from huggingface.co/models.", )
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help=(
                            "A folder containing the training data. Folder contents must follow the structure described in"
                            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"), )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
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

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")
    parser.add_argument("--optimizer_type",
        type=str,default="",
        help="Optimizer to use / オプティマイザの種類: AdamW (default), AdamW8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, AdaFactor",
    )
    # backward compatibility
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch) / Lionオプティマイザを使う（ lion-pytorch のインストールが必要）",
    )
    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") / オプティマイザの追加引数（例： "weight_decay=0.01 betas=0.9,0.999 ..."）',)
    parser.add_argument("--cache_dir", type=str,
                        default=None,
                        help="The directory where the downloaded models and datasets will be stored.", )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=[512, 768],
        help=(
            "res 512 for sd14, res 768 for sd21"
        ),
    )

    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Batch size (per device) for the training dataloader.")

    parser.add_argument("--num_train_epochs", type=int, default=100)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        required=True,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=6,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=[None, "wandb"],
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        required=True,
        help=(
            "Save a checkpoint of the training state every X updates."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        required=True,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None
    )

    # below are additional params
    parser.add_argument(
        "--token_loss_scale",
        type=float,
        required=True
    )

    parser.add_argument('--train_down', nargs='+', type=int, help='use which res layers in U-Net down', default=[])
    parser.add_argument('--train_mid', nargs='+', type=int, help='use which res layers in U-Net mid', default=[])
    parser.add_argument('--train_up', nargs='+', type=int, help='use which res layers in U-Net up', default=[])

    parser.add_argument("--pixel_loss_scale", type=float, required=True)
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--image_column", type=str, default="image")

    args = parser.parse_args()

    main(args)