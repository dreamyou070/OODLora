import importlib
import argparse
import re
import math
import os
import sys
import tempfile
from library import model_util
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
import torch
from torch import nn
import torch.nn.functional as F
from functools import lru_cache
from attention_store import AttentionStore
import copy
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

@lru_cache(maxsize=128)
def match_layer_name(layer_name:str, regex_list_str:str) -> bool:
    """
    Check if layer_name matches regex_list_str.
    """
    regex_list = regex_list_str.split(',')
    for regex in regex_list:
        regex = regex.strip() # remove space
        if re.match(regex, layer_name):
            return True
    return False
def register_attention_control(unet : nn.Module, controller:AttentionStore, mask_threshold:float=1) :
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
            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1],
                                                         key.shape[1], dtype=query.dtype,
                                                         device=query.device),
                                             query,key.transpose(-1, -2),beta=0,alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            if is_cross_attention:
                if trg_indexs_list is not None and mask is not None:
                    trg_indexs = trg_indexs_list
                    batch_num = len(trg_indexs)
                    attention_probs_batch = torch.chunk(attention_probs, batch_num, dim=0)
                    for batch_idx, attention_prob in enumerate(attention_probs_batch) :
                        batch_trg_index = trg_indexs[batch_idx] # two times
                        head_num = attention_prob.shape[0]
                        res = int(math.sqrt(attention_prob.shape[1]))
                        word_heat_map_list = []
                        for word_idx in batch_trg_index :
                            # head, pix_len
                            word_heat_map = attention_prob[:, :, word_idx]
                            word_heat_map_ = word_heat_map.reshape(-1, res, res)
                            word_heat_map_ = word_heat_map_.mean(dim=0)
                            word_heat_map_ = F.interpolate(word_heat_map_.unsqueeze(0).unsqueeze(0),
                                                           size=((512, 512)),mode='bicubic').squeeze()
                            word_heat_map_list.append(word_heat_map_)
                        word_heat_map_ = torch.stack(word_heat_map_list, dim=0) # (word_num, 512, 512)
                        # saving word_heat_map
                        # ------------------------------------------------------------------------------------------------------------------------------
                        # mask = [512,512]
                        mask_ = mask[batch_idx].to(attention_prob.dtype) # (512,512)
                        # thresholding, convert to 1 if upper than threshold else itself
                        mask_ = torch.where(mask_ > mask_threshold, torch.ones_like(mask_), mask_)
                        # check if mask_ is frozen, it should not be updated
                        assert mask_.requires_grad == False, 'mask_ should not be updated'
                        masked_heat_map = word_heat_map_ * mask_
                        attn_loss = F.mse_loss(word_heat_map_.mean(), masked_heat_map.mean())
                        controller.store(attn_loss, layer_name)
                # check if torch.no_grad() is in effect
                elif torch.is_grad_enabled(): # if not, while training, trg_indexs_list should not be None
                    if mask is None:
                        raise RuntimeError("mask is None but hooked to cross attention layer. Maybe the dataset does not contain mask properly.")
                    raise RuntimeError("trg_indexs_list is None but hooked to cross attention layer. Maybe the dataset does not contain trigger token properly.")

            hidden_states = torch.bmm(attention_probs, value)
            #if is_cross_attention :
            #    print(f'layer {layer_name} hidden_states.shape : {hidden_states.shape}')
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

class NetworkTrainer:

    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(self, args: argparse.Namespace, current_loss, avr_loss, lr_scheduler,
                           keys_scaled=None, mean_norm=None, maximum_norm=None, **kwargs):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}
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
    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, tokenizers, text_encoders, data_loader, weight_dtype ):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device)

    def extract_triggerword_index(self, input_ids):
        cls_token = 49406
        pad_token = 49407
        if input_ids.dim() == 3 :
            input_ids = torch.flatten(input_ids, start_dim=1)
        batch_num, sen_len = input_ids.size()
        batch_index_list = []

        for batch_index in range(batch_num) :
            token_ids = input_ids[batch_index, :].squeeze()
            index_list = []
            for index, token_id in enumerate(token_ids):
                if token_id != cls_token and token_id != pad_token :
                    index_list.append(index)
            batch_index_list.append(index_list)
        return batch_index_list

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizers[0], text_encoders[0], weight_dtype)
        return encoder_hidden_states

    def call_unet(self,args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype,
                  trg_indexs_list, mask_imgs):
        noise_pred = unet(noisy_latents, timesteps, text_conds, trg_indexs_list=trg_indexs_list, mask_imgs=mask_imgs, ).sample
        return noise_pred



    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet,
                      attention_storer=None,
                      efficient=False,
                      save_folder_name=None):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet,
                                 attention_storer=attention_storer,
                                 efficient=efficient,
                                 save_folder_name=save_folder_name)

    def get_input_ids(self, args, caption, tokenizer):
        tokenizer_max_length = args.max_token_length + 2
        # [1,77]
        input_ids = tokenizer(caption, padding="max_length", truncation=True, max_length=tokenizer_max_length, return_tensors="pt").input_ids
        if tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                for i in range( 1, tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2) :
                    ids_chunk = (input_ids[0].unsqueeze(0),input_ids[i : i + tokenizer.model_max_length - 2],  input_ids[-1].unsqueeze(0),)
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                for i in range(1, tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id
                    iids_list.append(ids_chunk)
            # [3,77]
            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids

    def train(self, args):

        print("\n step 1. preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process
        if args.log_with == 'wandb' and is_main_process:
            import wandb
            wandb.init(project=args.wandb_init_name,name=args.wandb_run_name)

        print("\n step 5. loading stable diffusion")
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
        _, text_encoder_org, vae_org, unet_org = self.load_target_model(args, weight_dtype, accelerator)
        text_encoders_org = text_encoder_org if isinstance(text_encoder_org, list) else [text_encoder_org]
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        print("\n step 7-1. prepare network")
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        pretrained_network_dir = args.output_dir
        network_dirs = os.listdir(pretrained_network_dir)
        for network_dir in network_dirs :
            if 'safetensors' in network_dir and 'last' in network_dir :
                network_file = os.path.join(pretrained_network_dir, network_dir)
                #print(f"load network from {network_file}")
                network_name, ext = os.path.splitext(network_dir)
                if 'epoch' in network_name :
                    epoch_info = int(network_name.split('-')[-1])
                elif 'last' in network_name :
                    epoch_info = 1000
                else :
                    epoch_info = -1
                    # learned network state dict
                from safetensors.torch import load_file
                weights_sd = load_file(network_file)
                layer_names = weights_sd.keys()
                efficient_layers = args.efficient_layer.split(",")
                #unefficient_layers = args.unefficient_layer.split(",")
                #save_folder_name = 'unefficient_' + '_'.join(efficient_layers)
                #save_folder_name = 'efficient_' +'_'.join(efficient_layers )
                save_folder_name = args.save_folder_name
                for layer_name in layer_names:
                    score = 0
                    for efficient_layer in efficient_layers:
                        if efficient_layer in layer_name:
                            if 'attn2_to_k' not in layer_name and 'attn2_to_v' not in layer_name:
                                score += 1
                    if score == 0 and 'alpha' not in layer_name:
                        weights_sd[layer_name] = weights_sd[layer_name] * 0

                    if 'alpha' in layer_name:
                        if 'text' in layer_name or 'attn2_to_k' in layer_name or 'attn2_to_v' in layer_name  :
                            weights_sd[layer_name] = weights_sd[layer_name] * 1
                        #if 'text' in layer_name or 'attn2_to_k' in layer_name or 'attn2_to_v' in layer_name or 'attn1_to_k' in layer_name or 'attn1_to_v' in layer_name :
                        #    weights_sd[layer_name] = weights_sd[layer_name] * 1
                        else :
                            weights_sd[layer_name] = weights_sd[layer_name] * 0

                        #print(f'layer_name : {layer_name} , alpha_value : {alpha_value}')
                    # because alpha is np, should be on cpu
                    #else :
                    #    print(f'layer to use : {layer_name}')
                    weights_sd[layer_name] = weights_sd[layer_name].to("cpu")
                # ------------------------------------------------------------------------------------------------------
                # 2) make empty network
                vae_copy,text_encoder_copy, unet_copy = copy.deepcopy(vae_org), copy.deepcopy(text_encoder_org).to("cpu" ), copy.deepcopy(unet_org)
                temp_network, weights_sd = network_module.create_network_from_weights(multiplier=1, file=None,block_wise=None,
                                                                                      vae=vae_copy, text_encoder=text_encoder_copy, unet=unet_copy,
                                                                                      weights_sd=weights_sd,for_inference=False,)
                text_encoder_loras = temp_network.text_encoder_loras
                for text_encoder_lora in text_encoder_loras :
                    lora_name = text_encoder_lora.lora_name
                    text_encoder_lora.lora_down.weight.data = weights_sd[f'{lora_name}.lora_down.weight']
                    text_encoder_lora.lora_up.weight.data = weights_sd[f'{lora_name}.lora_up.weight']
                    text_encoder_lora.to(weight_dtype).to(accelerator.device)
                unet_loras = temp_network.unet_loras
                for unet_lora in unet_loras :
                    lora_name = unet_lora.lora_name
                    unet_lora.lora_down.weight.data = weights_sd[f'{lora_name}.lora_down.weight']
                    unet_lora.lora_up.weight.data = weights_sd[f'{lora_name}.lora_up.weight']
                    unet_lora.to(weight_dtype).to(accelerator.device)

                # 3) to accelerator.devicef
                vae_copy.to(weight_dtype).to(accelerator.device)
                unet_copy.to(weight_dtype).to(accelerator.device)
                text_encoder_copy.to(weight_dtype).to(accelerator.device)
                # 4) applying to deeplearning network

                temp_network.apply_to(text_encoder_org,
                                      unet_org)
                self.sample_images(accelerator, args, epoch_info, 0, accelerator.device, vae_copy, tokenizer,
                                   text_encoder_copy, unet_copy, efficient=True, save_folder_name = save_folder_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--no_metadata", action="store_true",
                        help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）", )
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")
    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--network_dim", type=int, default=None,
                        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）")
    parser.add_argument("--network_alpha",type=float,default=1,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)",)
    parser.add_argument("--network_dropout",type=float,default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)",)
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value) / ネットワークへの追加の引数")
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
    parser.add_argument("--dim_from_weights",action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",)
    parser.add_argument("--scale_weight_norms",type=float,default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. ",)
    parser.add_argument("--base_weights",type=str,default=None,nargs="*",
                        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",)
    parser.add_argument("--base_weights_multiplier",type=float,default=None,nargs="*",
                        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",)
    parser.add_argument("--no_half_vae",action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",)
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    parser.add_argument("--wandb_init_name", type=str)
    parser.add_argument("--wandb_log_template_path", type=str)
    parser.add_argument("--wandb_key", type=str)
    parser.add_argument("--trg_concept", type=str, default='haibara')
    parser.add_argument("--net_key_names", type=str, default='text')
    parser.add_argument("--class_caption_dir", type=str, default='./sentence_datas/cat_sentence_100.txt' )
    # class_caption
    parser.add_argument("--class_caption", type=str, default='girl')
    parser.add_argument("--heatmap_loss", action='store_true')
    parser.add_argument("--attn_loss_ratio", type=float, default=1.0)
    parser.add_argument("--mask_dir", type=str)
    parser.add_argument("--pretraining_epochs", type=int, default = 10)
    # masked_loss
    parser.add_argument("--masked_loss", action='store_true')
    parser.add_argument("--only_second_training", action='store_true')
    parser.add_argument("--only_third_training", action='store_true')
    parser.add_argument("--first_second_training", action='store_true')
    parser.add_argument("--second_third_training", action='store_true')
    parser.add_argument("--first_second_third_training", action='store_true')
    parser.add_argument("--attn_loss_layers", type=str, default="all", help="attn loss layers, can be splitted with ',', matches regex with given string. default is 'all'")
    # mask_threshold (0~1, default 1)
    parser.add_argument("--mask_threshold", type=float, default=1.0, help="Threshold for mask to be used as 1")
    parser.add_argument("--heatmap_backprop", action = 'store_true')
    parser.add_argument('--class_token', default='cat', type=str)
    parser.add_argument('--unet_net_key_names', default='proj_in,ff_net', type=str)
    parser.add_argument("--te_freeze", action='store_true')
    parser.add_argument("--efficient_layer", type=str)
    parser.add_argument("--unefficient_layer", type=str)
    parser.add_argument("--save_folder_name", type=str)
    args = parser.parse_args()
    # overwrite args.attn_loss_layers if only_second_training, only_third_training, second_third_training, first_second_third_training is True
    if args.only_second_training:
        args.attn_loss_layers = 'down_blocks_2,up_blocks_1'
    elif args.only_third_training:
        args.attn_loss_layers = 'down_blocks_1,up_blocks_2'
    elif args.first_second_training:
        args.attn_loss_layers = 'mid,down_blocks_2,up_blocks_1'
    elif args.second_third_training:
        args.attn_loss_layers = 'down_blocks_2,up_blocks_1,down_blocks_1,up_blocks_2'
    elif args.first_second_third_training:
        args.attn_loss_layers = 'mid,down_blocks_2,up_blocks_1,down_blocks_1,up_blocks_2'
    else :
        args.attn_loss_layers = 'all'

    # if any of only_second_training, only_third_training, second_third_training, first_second_third_training is True, print message to notify user that args.attn_loss_layers is overwritten
    if args.only_second_training or args.only_third_training or args.second_third_training or args.first_second_third_training:
        print(f"args.attn_loss_layers is overwritten to {args.attn_loss_layers} because only_second_training, only_third_training, second_third_training, first_second_third_training is True")

    if args.wandb_init_name is not None:
        tempfile_new = tempfile.NamedTemporaryFile()
        print(f"Created temporary file: {tempfile_new.name}")
        if args.wandb_log_template_path is not None:
            with open(args.wandb_log_template_path, 'r', encoding='utf-8') as f:
                lines = f.read()
        else:
            lines = '''[wandb]
    name = "{0}"'''
        tempfile_path = tempfile_new.name
        with open(tempfile_path, 'w', encoding='utf-8') as f:
            # format
            f.write(lines.format(args.wandb_init_name))
        args.log_tracker_config = tempfile_path #overwrite
    args = train_util.read_config_from_file(args, parser)
    trainer = NetworkTrainer()
    trainer.train(args)

