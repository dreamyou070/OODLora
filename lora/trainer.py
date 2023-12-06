import os
import re
import toml
import subprocess
import argparse
import shutil
import json
import socket
from typing import List, Tuple
from ast import literal_eval
from image_util import process_path
import csv


def get_available_port(port: int, max_retries=100) -> int:
  tried_ports = []
  while port < 65535:
    try:
      sock = socket.socket()
      sock.bind(("127.0.0.1", port))
      sock.close()
      return port
    except (OSError, socket.error):
      print(f"Port {port} is already in use.")
      tried_ports.append(port)
      port += 1
    if len(tried_ports) > max_retries:
      break
  raise Exception("No available ports found!")


def validate_dataset():
  if override_dataset_config_file:
    print("Using custom dataset config file ", override_dataset_config_file)
    custom_dataset = override_dataset_config_file
  else:
    custom_dataset = None
  global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, keep_tokens_weight, weighted_captions, adjust_tags
  supported_types = (".png", ".jpg", ".jpeg", 'jfif')

  print("\n💿 Checking dataset...")
  if not project_name.strip() or any(c in project_name for c in " .()\"'\\/"):
    print("💥 Error: Please choose a valid project name.")
    raise Exception("Invalid project name")

  if custom_dataset:
    try:
      print("📄 Using custom dataset config file "+custom_dataset)
      assert os.path.exists(custom_dataset), "Custom dataset config file not found at "+custom_dataset
      datconf = toml.load(custom_dataset)
      datasets = [d for d in datconf["datasets"][0]["subsets"]]
    except Exception as e:
      print(f"💥 Error: Your custom dataset is invalid or contains an error! Please check the original template.")
      print(e)
      raise e
    reg = [d for d in datasets if d.get("is_reg", False)]
    for r in reg:
      print("📁"+r["image_dir"].replace("/content/drive/", "") + " (Regularization)")
    datasets = [d for d in datasets if d not in reg]
    datasets_dict = {d["image_dir"]: d["num_repeats"] for d in datasets}
    folders = datasets_dict.keys()
    files = [f for folder in folders for f in os.listdir(folder)]
    images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets_dict[folder]) for folder in folders}
  else:
    folders = [images_folder]
    files = os.listdir(images_folder)
    images_repeats = {images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), num_repeats)}

  for folder in folders:
    if not os.path.exists(folder):
      print(f"💥 Error: The folder {folder.replace('/content/drive/', '')} doesn't exist.")
      raise Exception("Empty folder")
  for folder, (img, rep) in images_repeats.items():
    if not img:
      print(f"💥 Error: Your {folder.replace('/content/drive/', '')} folder is empty.")
      raise Exception("Empty folder")
  for f in files:
    if not f.lower().endswith(".txt") and not f.lower().endswith(supported_types):
      print(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")
      raise Exception("Invalid file")

  if not [txt for txt in files if txt.lower().endswith(".txt")]:
    caption_extension = ""
  if continue_from_lora and not (continue_from_lora.endswith(".safetensors") and os.path.exists(continue_from_lora)):
    print(f"💥 Error: Invalid path to existing Lora. Example: /content/drive/MyDrive/Loras/example.safetensors")
    raise Exception("Invalid Lora path")

  pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
  steps_per_epoch = pre_steps_per_epoch/train_batch_size
  total_steps = max_train_steps or int(max_train_epochs*steps_per_epoch)
  estimated_epochs = int(total_steps/steps_per_epoch)
  lr_warmup_steps = int(total_steps*lr_warmup_ratio)

  for folder, (img, rep) in images_repeats.items():
    print("📁"+folder.replace("/content/drive/", ""))
    print(f"📈 Found {img} images with {rep} repeats, equaling {img*rep} steps.")
  print(f"📉 Divide {pre_steps_per_epoch} steps by {train_batch_size} batch size to get {steps_per_epoch} steps per epoch.")
  if max_train_epochs:
    print(f"🔮 There will be {max_train_epochs} epochs, for around {total_steps} total training steps.")
  else:
    print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

  if total_steps > 10000:
    print("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...")
    raise Exception("Too many steps")

  if adjust_tags:
    print(f"\n📎 Weighted tags: {'ON' if weighted_captions else 'OFF'}")
    if weighted_captions:
      print(f"📎 Will use {keep_tokens_weight} weight on {keep_tokens} activation tag(s)")
    print("📎 Adjusting tags...")
    adjust_weighted_tags(folders, keep_tokens, keep_tokens_weight, weighted_captions)

  return True

def adjust_weighted_tags(folders, keep_tokens: int, keep_tokens_weight: float, weighted_captions: bool):
  weighted_tag = re.compile(r"\((.+?):[.\d]+\)(,|$)")
  for folder in folders:
    for txt in [f for f in os.listdir(folder) if f.lower().endswith(".txt")]:
      with open(os.path.join(folder, txt), 'r') as f:
        content = f.read()
      # reset previous changes
      content = content.replace('\\', '')
      content = weighted_tag.sub(r'\1\2', content)
      if weighted_captions:
        # re-apply changes
        content = content.replace(r'(', r'\(').replace(r')', r'\)').replace(r':', r'\:')
        if keep_tokens_weight > 1:
          tags = [s.strip() for s in content.split(",")]
          for i in range(min(keep_tokens, len(tags))):
            tags[i] = f'({tags[i]}:{keep_tokens_weight})'
          content = ", ".join(tags)
      with open(os.path.join(folder, txt), 'w') as f:
        f.write(content)

def create_config():
  global dataset_config_file, config_file, model_file, port_num, port_fallback
  from accelerate.utils import write_basic_config
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["BITSANDBYTES_NOWELCOME"] = "1"
  os.environ["SAFETENSORS_FAST_GPU"] = "1"
  if os.path.exists(accelerate_config_file):
    # remove old config file
    os.remove(accelerate_config_file)
  write_basic_config(save_location=accelerate_config_file)
  # load config file and change settings
  if port_fallback:
    # search available port
    new_port_num = get_available_port(port_num)
    if new_port_num != port_num:
      print(f"\n⚠️ Port {port_num} is not available. Using port {new_port_num} instead.")
  else:
    new_port_num = port_num
  with open(accelerate_config_file, 'r') as configfile:
    config_json = json.load(configfile)
    config_json["main_process_port"] = new_port_num
  with open(accelerate_config_file, 'w') as configfile:
    json.dump(config_json, configfile, indent=2)
  if resume:
    resume_points = [f.path for f in os.scandir(output_folder) if f.is_dir()]
    resume_points.sort()
    last_resume_point = resume_points[-1] if resume_points else None
  else:
    last_resume_point = None

  if override_config_file:
    config_file = override_config_file
    print(f"\n⭕ Using custom config file {config_file}")
  else:
    config_dict = {
      "additional_network_arguments": {
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_module": network_module,
        "network_args": network_args,
        "network_train_unet_only": True if text_encoder_lr == 0 else None,
        "network_weights": continue_from_lora if continue_from_lora else None
      },
      "optimizer_arguments": {
        "learning_rate": unet_lr,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        "lr_warmup_steps": lr_warmup_steps if lr_scheduler != "constant" else None,
        "optimizer_type": optimizer,
        "optimizer_args": optimizer_args if optimizer_args else None,
      },
      "training_arguments": {
        "max_train_steps": max_train_steps,
        "max_train_epochs": max_train_epochs,
        "save_every_n_epochs": save_every_n_epochs,
        "save_last_n_epochs": keep_only_last_n_epochs,
        "train_batch_size": train_batch_size,
        "noise_offset": None,
        "clip_skip": clip_skip,
        "min_snr_gamma": min_snr_gamma_value,
        "weighted_captions": weighted_captions,
        "seed": training_seed,
        "max_token_length": 225,
        "xformers": xformers,
        "lowram": False,
        "max_data_loader_n_workers": 8,
        "persistent_data_loader_workers": True,
        "save_precision": precision_type,
        "mixed_precision": precision_type,
        "output_dir": output_folder,
        "logging_dir": log_folder,
        "output_name": project_name,
        "log_prefix": project_name,
        "save_state": save_state,
        "save_last_n_epochs_state": 1 if save_state else None,
        "resume": last_resume_point,
        "sample_every_n_steps": sample_num if sample_num and sample_opt.lower() == 'steps' else None,
        "sample_every_n_epochs": sample_num if sample_num and sample_opt.lower() == 'epoch' else None,
        "sample_prompts": prompt_path if prompt_path and sample_opt.lower() != 'None' else None,
        "max_grad_norm": max_grad_norm if max_grad_norm else 0.0
      },
      "model_arguments": {
        "pretrained_model_name_or_path": model_file,
      },
      "saving_arguments": {
        "save_model_as": "safetensors",
      },
      "dreambooth_arguments": {
        "prior_loss_weight": 1.0,
      },
      "dataset_arguments": {
        "cache_latents": True if not flip_aug and not color_aug else False,
      },
      "extra_arguments": extra_args_dict
    }

    for key in config_dict:
      if isinstance(config_dict[key], dict):
        config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

    with open(config_file, "w") as f:
      f.write(toml.dumps(config_dict))
    print(f"\n📄 Config saved to {config_file}")

  if override_dataset_config_file:
    # copy to dataset_config_file
    shutil.copyfile(override_dataset_config_file, dataset_config_file)
    dataset_config_file = override_dataset_config_file
    print(f"⭕ Using custom dataset config file {dataset_config_file}")
  else:
    dataset_config_dict = {
      "general": {
        "resolution": resolution,
        "shuffle_caption": shuffle_caption,
        "keep_tokens": keep_tokens,
        "flip_aug": flip_aug,
        "caption_extension": caption_extension,
        "enable_bucket": True,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
        "min_bucket_reso": 320 if resolution > 640 else 256,
        "max_bucket_reso": 1280 if resolution > 640 else 1024,
      },
      "datasets": toml.load(custom_dataset)["datasets"] if custom_dataset else [
        {
          "subsets": [
            {
              "num_repeats": num_repeats,
              "image_dir": images_folder,
              "class_tokens": None if caption_extension else project_name
            }
          ]
        }
      ]
    }

    for key in dataset_config_dict:
      if isinstance(dataset_config_dict[key], dict):
        dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

    with open(dataset_config_file, "w") as f:
      f.write(toml.dumps(dataset_config_dict))
    print(f"📄 Dataset config saved to {dataset_config_file}")

def main():
  for dir in (main_dir, repo_dir, log_folder, images_folder, output_folder, config_folder):
    if not dir:
      continue
    print("Creating directory "+os.path.abspath(dir))
    os.makedirs(dir, exist_ok=True)
  if not validate_dataset():
    return
  assert os.path.exists(model_file), "Model file not found at "+model_file
  create_config()
  assert os.path.exists(accelerate_config_file), "Config file not found at "+accelerate_config_file
  print("\n⭐ Starting trainer...\n")
  os.chdir(repo_dir)
  print("Running accelerate launch... at "+repo_dir)
  # call .\venv\Scripts\activate.bat
  # set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib
    # windows, required user venv..
  command_list = ["accelerate", "launch", "--config_file="+accelerate_config_file, "--num_cpu_threads_per_process=1", "train_network.py", "--dataset_config="+dataset_config_file, "--config_file="+config_file]
  with open(os.path.join(log_folder, "accelerate_launch_commands_log.txt"), "a") as f:
    f.write(" ".join(command_list))
    
  subprocess.check_call(command_list,
                        shell=True if os.name == 'nt' else False)
  # move model to output folder

def get_separated_list(result_list) -> dict:
    all_list = [1] * 25
    if len(result_list) == 17:
        # using minimal-setup which is being applied for loras
        text_encoder_lr = result_list[0]
        result_list = result_list[1:]
    else:
        text_encoder_lr = 1
    if len(result_list) == 16:
        convert_target_idxs = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        for idx, value in enumerate(result_list):
            target_idx = convert_target_idxs[idx]
            all_list[target_idx] = value
    else:
        # using full-setup which is being applied for loras
        text_encoder_lr = 1
        for _idx, _items in enumerate(result_list):
            all_list[_idx] = _items
    # 13th item(idx 12) is middle
    down, middle, up = all_list[:12], all_list[12], all_list[13:]

    return text_encoder_lr, down, middle, up

def add_basic_arguments(parser : argparse.ArgumentParser) -> List[str]:
  """
  Adds basic arguments to the parser.
  """
  parser.add_argument('--cuda_device', type=str, default='4',
                      help='CUDA device number (default: 4)')
  parser.add_argument('--project_name_base', type=str, default='WEBTOON_ALL',
                      help='Base name for the project (default: WEBTOON_ALL)')
  parser.add_argument('--model_file', type=str, default='./model.ckpt',
                      help='Path to the model file (default: ./model.ckpt)')
  parser.add_argument('--custom_dataset', type=str, default="",
                      help='Custom dataset config path. (default: None)')
  # add image_folder
  parser.add_argument('--images_folder', type=str, default='')
  #add repo_dir
  parser.add_argument('--repo_dir', type=str, default='.')
  # add custon suffix
  parser.add_argument('--custom_suffix', type=str, default='',
                      help='Custom suffix for the project name (default: "")')
  parser.add_argument('--target_path', type=str, default='',
                      help='Target path for the project (default: "")')
  parser.add_argument('--temp_dir', type=str, default='', help='Temporary directory for the project (default: "")')
  # add port to use for accelerate
  parser.add_argument('--port', type=int, default=20060, help='Port to use for accelerate (default: 20060)')
  # should we use port fallback
  parser.add_argument('--port_fallback', type=bool, default=True, help='Use port fallback (default: False)')
  return []

def add_sample_args(parser : argparse.ArgumentParser) -> List[str]:
  """
  Adds sample arguments to the parser.
  """
  # prompt
  parser.add_argument('--prompt_path', type=str, default='', help='Prompt file for the project (default: ""), ex : `character is haibara ai,1girl --s 16 --w 512 --h 768 --d 42`')
  # sample_opt ('epoch', 'step')
  parser.add_argument('--sample_opt', type=str, default='epoch', help='Sample option for the project (default: epoch, can be None)')
  # sample_num
  parser.add_argument('--sample_num', type=int, default=1, help='Sample number for the project (default: 1)')
  return []

def add_lora_args(parser : argparse.ArgumentParser) -> List[str]:
  """
  Adds lora arguments to the parser.
  """
  parser.add_argument('--network_dim', type=int, default=16,
                      help='Dimension of the network (default: 16)')
  parser.add_argument('--network_alpha', type=int, default=8,
                      help='Alpha value for the network (default: 8)')
  parser.add_argument('--conv_dim', type=int, default=8,
                      help='Dimension of the convolutional layers (default: 8)')
  parser.add_argument('--conv_alpha', type=int, default=1,
                      help='Alpha value for the convolutional layers (default: 1)')
  parser.add_argument('--lora_type', type=str, default='LoRA', help='LoRA type for the project (default: LoRA)')
  return []

def add_training_args(parser : argparse.ArgumentParser) -> List[str]:
  """
  Adds training arguments to the parser.
  """
  parser.add_argument('--optimizer', type=str, default='AdamW8bit',
                      help='Optimizer to use (default: AdamW8bit)')
  parser.add_argument('--num_repeats', type=int, default=10,
                      help='Number of repeats (default: 10)')
  parser.add_argument('--epoch_num', type=int, default=10,
                      help='Number of epochs (default: 40)')
  parser.add_argument('--train_batch_size', type=int, default=4,
                      help='Batch size for training (default: 4)')
  parser.add_argument('--unet_lr', type=float, default=1e-4,
                      help='Learning rate for the UNet (default: 1e-4)')
  parser.add_argument('--text_encoder_lr', type=float, default=2e-5,
                      help='Learning rate for the text encoder (default: 2e-5)')
  parser.add_argument('--lr_scheduler', type=str, default='cosine_with_restarts', help='LR scheduler for the project (default: cosine_with_restarts)')
  # xformers
  parser.add_argument('--xformers', type=str, default='False', help='Xformers for the project (default: False)')
  # precision_type
  parser.add_argument('--precision_type', type=str, default='bf16', help='Precision type for the project (default: bf16, available: bf16, fp32, fp16)')
  return []

def add_regularization_args(parser : argparse.ArgumentParser) -> List[str]:
  """
  Adds regularization arguments to the parser.
  """
  parser.add_argument('--max_grad_norm', type=float, default=0.0, help='Max grad norm for the project (default: 0.0)')
  parser.add_argument('--up_lr_weight', type=str, default="[1,1,1,1,1,1,1,1,1,1,1,1]",
                      help='Lora block weight up for the project (default: [1,1,1,1,1,1,1,1,1,1,1,1])')
  parser.add_argument('--down_lr_weight', type=str, default="[1,1,1,1,1,1,1,1,1,1,1,1]",
                      help='Lora block weight down for the project (default: [1,1,1,1,1,1,1,1,1,1,1,1])')
  parser.add_argument('--mid_lr_weight', type=str, default='1',
                      help='Lora block weight mid for the project (default: 1)')
  # add 17-length list input for text/down/middle/up lr weight parsing
  parser.add_argument('--lbw_weights', type=str, default='',
                      help='Optional 16 or 17-length list input for text/down/middle/up lr weight parsing (default: ""), this will disable other block lr arguments')
  return []

def add_gor_args(parser: argparse.ArgumentParser)-> List[str]:
    parser.add_argument("--gor_num_groups", type=int, default=32, help="number of groups for group orthogonality regularization")
    parser.add_argument("--gor_regularization_type", type=str, default=None, help="type of group orthogonality regularization, 'inter' or 'intra'")
    parser.add_argument("--gor_name_to_regularize", type=str, default='up_blocks.*_lora\.up', help="name of the layer to regularize, e.g. 'up_blocks.*_lora\.up'")
    parser.add_argument("--gor_regularize_fc_layers", type=str, default="True", help="whether to regularize fully connected layers")
    parser.add_argument("--gor_ortho_decay", type=float, default=1e-6, help="decay for group orthogonality regularization")
    parser.add_argument("--gor_regularization", type=str, default="False", help="whether to enable group orthogonality regularization")
    return ['gor_num_groups', 'gor_regularization_type', 'gor_name_to_regularize', 'gor_regularize_fc_layers', 'gor_ortho_decay', 'gor_regularization']



def add_augmentation_args(parser : argparse.ArgumentParser) -> List[str]:
  # face_crop_aug_range [float, float]
  # flip_aug bool
  # color_aug bool
  # random_crop bool
  """
  Adds augmentation arguments to the parser.
  """
  parser.add_argument('--face_crop_aug_range', type=str, default="0.0,0.0",
                      help='Face crop augmentation range for the project (default: 0.0, 0.0)')
  parser.add_argument('--flip_aug', type=str, default=False,
                      help='Flip augmentation for the project (default: False)')
  parser.add_argument('--color_aug', type=str, default=False,
                      help='Color augmentation for the project (default: False)')
  parser.add_argument('--random_crop', type=str, default=False,
                      help='Random crop for the project (default: False)')
  return ['face_crop_aug_range', 'flip_aug', 'color_aug', 'random_crop']

def add_extra_args(parser : argparse.ArgumentParser) -> List[str]:
  """
  Adds extra arguments to the parser.
  """
  # seed
  parser.add_argument('--seed', type=int, default=42, help='Seed for the project (default: 42)')
  # keep tokens
  parser.add_argument('--keep_tokens', type=int, default=1, help='Keep tokens for the project (default: 1)')
  # resolution
  parser.add_argument('--resolution', type=int, default=512, help='Resolution for the project (default: 512)')
  # clip skip
  parser.add_argument('--clip_skip', type=int, default=2, help='Clip skip for the project (default: 2)')
  # min snr gamma and min snr gamma value
  parser.add_argument('--min_snr_gamma', type=bool, default=True, help='Min snr gamma for the project (default: True)')
  # min snr gamma
  parser.add_argument('--min_snr_gamma_value', type=float, default=5.0, help='Min snr gamma value for the project (default: 5.0)')
  # zero_terminal_snr, it has to be string because bug
  parser.add_argument('--zero_terminal_snr', type=str, default='False', help='Zero terminal snr for the project (default: False)')
  # is_sd2
  parser.add_argument('--v2', type=str, default=False, help='Is sd2 for the project (default: False)')
  # v_parameterization
  parser.add_argument('--v_parameterization', type=str, default=False, help='V parameterization for the project (default: False)')
  return ['zero_terminal_snr', 'v2', 'v_parameterization']

def add_optimizer_args(parser : argparse.ArgumentParser) -> List[str]:
  """
  Adds optimizer arguments to the parser.
  """
  # betas 0.9, 0.999
  # eps 1e-8
  # weight_decay 0.01
  parser.add_argument('--adamw_betas', type=str, default='0.9,0.999',
                      help='Betas for the optimizer (default: 0.9,0.999)')
  parser.add_argument('--adamw_eps', type=float, default=1e-8,
                      help='Epsilon for the optimizer (default: 1e-8)')
  parser.add_argument('--adamw_weight_decay', type=float, default=0.01,
                      help='Weight decay for the optimizer (default: 0.01)')
  return []

def fix_boolean_args(argument_dict : dict) -> dict:
  """
  Fixes boolean arguments in the argument dict.
  """
  for key, value in argument_dict.items():
    if isinstance(value, str):
      if value.lower() == 'true':
        argument_dict[key] = True
      elif value.lower() == 'false':
        argument_dict[key] = False
      elif value.lower() == 'none':
        argument_dict[key] = None
  return argument_dict

def add_logging_args(parser: argparse.ArgumentParser) -> List[str]:
  # log_with : 'wandb', 'tensorboard', 'all', 'none'
  # wandb_api_key : str
  parser.add_argument('--log_with', type=str, default='all', help='Log with for the project (default: None)')
  parser.add_argument('--wandb_api_key', type=str, default='', help='Wandb api key for the project (default: "")')
  #log_tracker_config : path to config file, default : None
  parser.add_argument('--log_tracker_config', type=str, default='none', help='Log tracker config for the project (default: "none")')
  return ['log_with', 'wandb_api_key', 'log_tracker_config']
      
if __name__ == "__main__":
  extra_args = []
  parser = argparse.ArgumentParser(description='LoRA Trainer')
  extra_args.extend(add_basic_arguments(parser))
  extra_args.extend(add_sample_args(parser)) # prompt, sample_opt, sample_num
  extra_args.extend(add_lora_args(parser)) # network_dim, network_alpha, conv_dim, conv_alpha
  extra_args.extend(add_training_args(parser)) # optimizer, num_repeats, epoch_num, train_batch_size, unet_lr, text_encoder_lr, lr_scheduler
  extra_args.extend(add_regularization_args(parser)) # max_grad_norm, up_lr_weight, down_lr_weight, mid_lr_weight
  extra_args.extend(add_augmentation_args(parser)) # face_crop_aug_range, flip_aug, color_aug, random_crop
  extra_args.extend(add_gor_args(parser)) # group orthogonality regularization
  extra_args.extend(add_extra_args(parser)) # seed, keep_tokens, resolution, clip_skip 
  extra_args.extend(add_optimizer_args(parser)) # betas, eps, weight_decay
  extra_args.extend(add_logging_args(parser)) # log_with, wandb_api_key
  # keep tokens should be used only if first tags(comma separated) are properly tagged and set up in dataset.

  args = parser.parse_args()

  extra_args_dict = {
    k : getattr(args, k) for k in extra_args
  }
  extra_args_dict = fix_boolean_args(extra_args_dict) # fix boolean arguments
  
  print("Extra args dict : ", extra_args_dict)
  
  try:
    down_lr_weight = literal_eval(args.down_lr_weight)
    up_lr_weight = literal_eval(args.up_lr_weight)
    mid_lr_weight = float(args.mid_lr_weight)
  except Exception as e:
    print("Error while parsing lora block lr weights")
    raise e
  if args.lbw_weights != '':
    print("Using custom lora block lr weights")
    lbw_list = literal_eval(args.lbw_weights)
    if len(lbw_list) not in [16, 17]:
      raise Exception("Length of lbw_weights must be 16 or 17, but given "+str(len(lbw_list)))
    lbw_text_encoder_lr_mult, down_lr_weight, mid_lr_weight, up_lr_weight = get_separated_list(lbw_list)
  else:
    lbw_text_encoder_lr_mult = 1
  
  # TODO : separate validation, please clean up this mess
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
  if args.target_path != '':
    print("Target path will be used as "+args.target_path)
  project_name_base = args.project_name_base
  model_file = args.model_file
  optimizer = args.optimizer
  network_dim = args.network_dim
  network_alpha = args.network_alpha
  conv_dim = args.conv_dim
  training_seed = args.seed
  conv_alpha = args.conv_alpha
  num_repeats = args.num_repeats
  epoch_num = args.epoch_num
  sample_num = args.sample_num
  sample_opt = args.sample_opt
  prompt_path = args.prompt_path
  train_batch_size = args.train_batch_size
  unet_lr = args.unet_lr
  text_encoder_lr = args.text_encoder_lr * lbw_text_encoder_lr_mult
  suffix = args.custom_suffix
  max_grad_norm = args.max_grad_norm
  clip_skip = args.clip_skip
  color_aug = args.color_aug
  
  xformers = args.xformers
  precision_type = args.precision_type
  
  # parse optimizer args
  betas = args.adamw_betas
  # assert , in betas and length is 2 and both are float, such as 0.9,0.999
  betas = [float(s) for s in betas.split(',')]
  assert len(betas) == 2, "Betas must be length 2, but given "+str(len(betas))
  assert isinstance(betas[0], float) and isinstance(betas[1], float), "Betas must be float, but given "+str(betas)
  epsilon = args.adamw_eps
  weight_decay = args.adamw_weight_decay
  
  assert args.sample_opt in ['epoch', 'step', 'None'], "Sample option must be 'epoch' or 'step' or 'None', but given "+args.sample_opt
  assert args.sample_opt == 'None' or args.sample_num > 0, "Sample number must be positive, but given "+str(args.sample_num)
  assert sample_opt == 'None' or not prompt_path or os.path.exists(prompt_path), "Prompt file not found at "+prompt_path
  assert clip_skip in [1, 2], "Clip skip must be 1 or 2, but given "+str(clip_skip)
  assert max_grad_norm >= 0.0, "Max grad norm must be positive, but given "+str(max_grad_norm)
  curdir = os.path.dirname(os.path.abspath(__file__)) 
  root_dir = args.temp_dir if args.temp_dir != '' else "./Loras"
  os.chdir(curdir) # change directory to current directory
  repo_dir = args.repo_dir
  port_num = args.port
  port_fallback = args.port_fallback
  images_folder = f"./Loras/{project_name_base}/dataset/" if args.images_folder == '' else args.images_folder
  custom_dataset = None
  override_dataset_config_file = ""
  if args.custom_dataset and os.path.exists(args.custom_dataset):
    print("Custom dataset will be loaded from " + args.custom_dataset + " , images_folder will be ignored.")
    images_folder = ''
    custom_dataset = None
    override_dataset_config_file = args.custom_dataset 
  elif args.custom_dataset and not os.path.exists(args.custom_dataset):
    if args.custom_dataset.lower() == 'none' or args.custom_dataset.lower() == 'null' or args.custom_dataset.lower() == 'false':
      print("Custom dataset will not be loaded, images_folder will be used.")
      custom_dataset = None
      override_dataset_config_file = ""
    else:
      raise Exception("Custom dataset not found at " + args.custom_dataset)
  project_name = f"{project_name_base}_autotrain" #@param {type:"string"}
  skip_model_test = True
  assert not override_dataset_config_file or os.path.exists(override_dataset_config_file), "Custom dataset config file not found at "+override_dataset_config_file
  assert args.lr_scheduler in ['constant', 'cosine', 'cosine_with_restarts', 'constant_with_warmup', 'linear', 'polynomial'], "LR scheduler must be 'constant', 'cosine', 'cosine_with_restarts', 'constant_with_warmup', 'linear', or 'polynomial', but given "+args.lr_scheduler
  override_config_file = None
  optimizer_args = None
  continue_from_lora = ""
  weighted_captions = True ## True로 하면 Weighted Caption 적용
  adjust_tags = True
  keep_tokens_weight = 1.0 
  resolution = args.resolution
  flip_aug = False #@param {type:"boolean"}
  caption_extension = ".txt" #param {type:"string"}
  shuffle_tags = True #@param {type:"boolean"}
  shuffle_caption = shuffle_tags
  keep_tokens = args.keep_tokens #param {type:"slider", min:0, max:10, step:1}
  preferred_unit = "Epochs" #@param ["Epochs", "Steps"]
  max_train_epochs = epoch_num if preferred_unit == "Epochs" else None
  max_train_steps = epoch_num if preferred_unit == "Steps" else None
  save_every_n_epochs = 1 #@param {type:"number"}
  keep_only_last_n_epochs = False
  if not save_every_n_epochs:
    save_every_n_epochs = max_train_epochs
  if not keep_only_last_n_epochs:
    keep_only_last_n_epochs = max_train_epochs

  lr_scheduler = args.lr_scheduler #@param ["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"]
  lr_scheduler_number = 3 #@param {type:"number"}
  lr_scheduler_num_cycles = lr_scheduler_number if lr_scheduler == "cosine_with_restarts" else 0
  lr_scheduler_power = lr_scheduler_number if lr_scheduler == "polynomial" else 0
  lr_warmup_ratio = 0.05 #@param {type:"slider", min:0.0, max:0.5, step:0.01}
  lr_warmup_steps = 0
  min_snr_gamma = args.min_snr_gamma #@param {type:"boolean"}
  min_snr_gamma_value = args.min_snr_gamma_value if min_snr_gamma else None
  lora_type = args.lora_type #@param ["LoRA", "LoCon Lycoris", "LoHa Lycoris"]
  conv_compression = False #@param {type:"boolean"}
  network_module = "lycoris.kohya" if "Lycoris" in lora_type else "networks.lora"
  network_args = [
    "down_lr_weight="+','.join(str(x) for x in down_lr_weight),
    "up_lr_weight="+','.join(str(x) for x in up_lr_weight),
    "mid_lr_weight="+str(mid_lr_weight),
  ]
  if lora_type != "LoRA":
    network_args.append(f"conv_dim={conv_dim}")
    network_args.append(f"conv_alpha={conv_alpha}")
  if "Lycoris" in lora_type:
    network_args.append(f"algo={'loha' if 'LoHa' in lora_type else 'lora'}")
    network_args.append(f"disable_conv_cp={str(not conv_compression)}")
  save_state = False #param {type:"boolean"}
  resume = False #param {type:"boolean"}


  if optimizer == "DAdaptation":
    print("Using DAdaptation optimizer, the following arguments will be ignored:")
    print("unet_lr, text_encoder_lr, optimizer_args, lr_scheduler")
    optimizer_args = ["decouple=True",f"weight_decay={weight_decay}",f"betas=[{betas}]",f"eps={epsilon}"]
    unet_lr = 0.5
    text_encoder_lr = 0.5 * lbw_text_encoder_lr_mult
    if lr_scheduler != "constant_with_warmup":
      print(f"Using lr scheduler {lr_scheduler}, recommended for DAdaptation is constant_with_warmup")
    #network_alpha = network_dim
  elif optimizer == "AdamW8bit":
    optimizer_args = [f"betas={betas}", f"weight_decay={weight_decay}", f"eps={epsilon}"]
  
  main_dir      = root_dir
  log_folder    = os.path.join(main_dir, "_logs")
  config_folder = os.path.join(main_dir, project_name + suffix)
  output_folder = os.path.join(main_dir, project_name + suffix, "output")
  sample_folder = os.path.join(main_dir, project_name + suffix, "output", "sample")

  config_file = os.path.join(config_folder, "training_config.toml")
  dataset_config_file = os.path.join(config_folder, "dataset_config.toml")
  accelerate_config_file = os.path.join(config_folder, "accelerate_config/config.yaml")

  main()
  #after training, from output_folder, move model file to target_path
  target_path = args.target_path
  if target_path != '' and os.path.exists(output_folder):
    print("Moving model to target path " + target_path)
  if not os.path.exists(target_path):
    os.makedirs(target_path)
  #   # move {project_name}.safetensors to target_path
  #   # rename to {project_name} _suffix.safetensors
    os.rename(os.path.join(output_folder, f"{project_name}.safetensors"), os.path.join(output_folder, f"{project_name}_{suffix}.safetensors"))
    shutil.copy(os.path.join(output_folder, f"{project_name}_{suffix}.safetensors"), target_path)
    # copy config files with suffix
    shutil.copy(config_file, os.path.join(target_path, f"training_config_{suffix}.toml"))
    #shutil.copy(dataset_config_file, os.path.join(target_path, f"dataset_config_{suffix}.toml"))
    #shutil.copy(accelerate_config_file, os.path.join(target_path, f"accelerate_config_{suffix}.yaml"))
  result_samples = process_path(sample_folder)
  # move processed samples to target_path
  if not os.path.exists(os.path.join(target_path, "grid_samples")):
    os.makedirs(os.path.join(target_path, "grid_samples"))
  for grid_path in result_samples:
    prefix = f's{suffix}_'
    # move to target_path/samples/prefix+basename(grid_path)
    shutil.move(grid_path, os.path.join(target_path, "grid_samples", project_name + prefix + os.path.basename(grid_path)))
  

  print("Done!")
