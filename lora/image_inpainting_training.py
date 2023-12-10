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
import library.config_util as config_util
from library.config_util import (ConfigSanitizer,BlueprintGenerator,)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (apply_snr_weight,get_weighted_text_embeddings,prepare_scheduler_for_custom_training,
                                            scale_v_prediction_loss_like_noise_prediction,add_v_prediction_like_loss,)
from setproctitle import *
from diffusers import StableDiffusionInpaintPipeline
import torch, os
from diffusers import UNet2DConditionModel
import argparse

def main(args):

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


    print(f'\n step 2. make model')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(args.pretrained_dir)
    tokenizer = pipe.tokenizer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    unet_config_dir = '/data7/sooyeon/pretrained_stable_diffusion/models--runwayml--stable-diffusion-inpainting/snapshots/51388a731f57604945fddd703ecb5c50e8e7b49d/unet/config.json'
    unet_config = UNet2DConditionModel.load_config(pretrained_model_name_or_path=unet_config_dir)
    unet = UNet2DConditionModel.from_config(unet_config)

    print(f'\n step 3. dataset')
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
    user_config = {"datasets": [
        {"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)}]}
    """
    
    
    blueprint = blueprint_generator.generate(user_config,args,tokenizer=tokenizer)
    train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    """
    save_folder = '/data7/sooyeon/MyData/anomaly_detection/MVTecAD/bagel/paired_data_save'
    os.makedirs(save_folder, exist_ok=True)
    base_folder = '/data7/sooyeon/MyData/anomaly_detection/MVTecAD/bagel/paired_data'
    classes = os.listdir(base_folder)
    for class_name in classes:
        class_dir = os.path.join(base_folder, class_name)
        save_class_dir = os.path.join(save_folder, f'10_{class_name}')
        os.makedirs(save_class_dir, exist_ok=True)
        answer_dir = os.path.join(class_dir, 'perfect_rgb')
        origin_dir = os.path.join(class_dir, 'rgb')
        answers = os.listdir(answer_dir)
        for answer in answers:
            answer_path = os.path.join(answer_dir, answer)
            origin_path = os.path.join(origin_dir, answer)
            print(answer_path, origin_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    parser.add_argument('--process_title', type=str, default='parksooyeon')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--logging_dir",type=str,default=None,)
    parser.add_argument("--log_with",type=str,default='wandb',choices=["tensorboard", "wandb", "all"],)
    parser.add_argument("--log_prefix", type=str, default=None,
                        help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列")
    parser.add_argument('--wandb_api_key', type=str, default='3a3bc2f629692fa154b9274a5bbe5881d47245dc')
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数",)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="use mixed precision / 混合精度を使う場合、その精度")
    parser.add_argument('--output_dir', type=str, default='wandb')
    parser.add_argument('--wandb_init_name', type=str, default='wandb')
    # step 2. make model
    parser.add_argument('--pretrained_dir', type=str, default='/data7/sooyeon/pretrained_stable_diffusion/models--runwayml--stable-diffusion-inpainting')
    parser.add_argument('--train_data_dir', type=str,
                        default='/data7/sooyeon/pretrained_stable_diffusion/models--runwayml--stable-diffusion-inpainting')
    parser.add_argument('--reg_data_dir', type=str,default = None)
    args = parser.parse_args()
    main(args)