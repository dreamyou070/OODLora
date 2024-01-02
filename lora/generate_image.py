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
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None


def register_attention_control(unet_model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, ):
            is_cross = encoder_hidden_states is not None

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller.save(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)
            # all drop out in diffusers are 0.0
            # so we here ignore dropout

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    assert controller is not None, "controller must be specified"

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    down_count = 0
    up_count = 0
    mid_count = 0

    cross_att_count = 0
    sub_nets = unet_model.named_children()
    for net in sub_nets:

        if "down" in net[0]:
            down_temp = register_recr(net[1], 0, "down")
            cross_att_count += down_temp
            down_count += down_temp
        elif "up" in net[0]:
            up_temp = register_recr(net[1], 0, "up")
            cross_att_count += up_temp
            up_count += up_temp
        elif "mid" in net[0]:
            mid_temp = register_recr(net[1], 0, "mid")
            cross_att_count += mid_temp
            mid_count += mid_temp
    controller.num_att_layers = cross_att_count

def get_cross_attn_map_from_unet(attention_store: AttentionStore, reses=[64, 32, 16, 8], poses=["down", "mid", "up"]):

    #attention_maps = attention_store.get_average_attention()
    attention_maps = attention_store.step_store
    attn_dict = {}
    for pos in poses:
        for res in reses:
            temp_list = []
            for item in attention_maps[f"{pos}_cross"]:
                if item.shape[1] == res ** 2:
                    cross_maps = item.reshape(-1, res, res, item.shape[-1])
                    temp_list.append(cross_maps)
            # if such resolution exists
            if len(temp_list) > 0:
                attn_dict[f"{pos}_{res}"] = temp_list # length 1 or 3
    return attn_dict

def get_grounding_loss_by_layer(_gt_seg_list,
                                word_token_idx_ls,
                                res,                # 64,32,16,8
                                input_attn_map_ls,):
    gt_seg_list = deepcopy(_gt_seg_list)

    # reszie gt seg map to the same size with attn map
    resize_transform = transforms.Resize((res, res))
    noun_num = len(gt_seg_list)
    for i in range(len(gt_seg_list)):
        gt_seg_list[i] = resize_transform(gt_seg_list[i])
        gt_seg_list[i] = gt_seg_list[i].squeeze(0) # 1, 1, res, res => 1, 1, res(8,16,32,64), res(8,16,32,64)
        # add binary
        binary = (gt_seg_list[i] > 0.0).float() # 1, res, res
        gt_seg_list[i] = (gt_seg_list[i] > 0.0).float()

    ################### token loss start ###################
    # Following code is adapted from
    # https://github.com/silent-chen/layout-guidance/blob/08b687470f911c7f57937012bdf55194836d693e/utils.py#L27
    token_loss = 0.0
    for attn_map in input_attn_map_ls:
        # len is 3 or 1
        b, H, W, j = attn_map.shape
        for i in range(len(word_token_idx_ls)): # [[word1 token_idx1, word1 token_idx2, ...], [word2 token_idx1, word2 token_idx2, ...]]
            obj_loss = 0.0
            single_word_idx_ls = word_token_idx_ls[i] #[token_idx1, token_idx2, ...]
            mask = gt_seg_list[i]
            for obj_position in single_word_idx_ls:
                # ca map obj shape 8 * 16 * 16
                ca_map_obj = attn_map[:, :, :, obj_position].reshape(b, H, W) # 1, 8, 8
                trg_score =  (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)
                all_score =  ca_map_obj.reshape(b, -1).sum(dim=-1)
                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                obj_loss += (1.0 - torch.mean(activation_value)) ** 2
            token_loss += (obj_loss/len(single_word_idx_ls))
    # normalize with len words
    token_loss = token_loss / len(word_token_idx_ls)
    ################## token loss end ##########################

    ################## pixel loss start ######################
    # average cross attention map on different layers
    avg_attn_map_ls = []
    # input_attn_map_list ?
    for i in range(len(input_attn_map_ls)):
        # len is 1 or 3
        org_map = input_attn_map_ls[i] # head, res, res, 77
        map = input_attn_map_ls[i].reshape(-1, res, res, input_attn_map_ls[i].shape[-1]).mean(0) # res,res,77
        avg_attn_map_ls.append(map)
    avg_attn_map = torch.stack(avg_attn_map_ls, dim=0) # [ (8,8,77), (8,8,77)]
    avg_attn_map = avg_attn_map.sum(0) / avg_attn_map.shape[0] # res,res,77
    avg_attn_map = avg_attn_map.unsqueeze(0) # 1, rse,res, 77

    bce_loss_func = nn.BCELoss()
    pixel_loss = 0.0
    for i in range(len(word_token_idx_ls)):

        # token idx
        word_cross_attn_ls = []
        for token_idx in word_token_idx_ls[i]:
            # 2
            # 9
            # 5
            word_map = avg_attn_map[..., token_idx] # 1, res, res, 1
            word_cross_attn_ls.append(word_map)

        word_cross_attn_ls = torch.stack(word_cross_attn_ls, dim=0).sum(dim=0) # 1, rse,res
        print(f'word_cross_attn_ls.shape: {word_cross_attn_ls.shape}')
        print(f'gt_seg_list[i].shape: {gt_seg_list[i].shape}')

        pixel_loss += bce_loss_func(word_cross_attn_ls, gt_seg_list[i])

    # average with len word_token_idx_ls
    pixel_loss = pixel_loss / len(word_token_idx_ls)
    ################## pixel loss end #########################

    return {
        "token_loss" : token_loss,
        "pixel_loss": pixel_loss,
    }

def main(args) :

    parent = os.path.split(args.network_weights)[0]
    folder = os.path.split(parent)[-1]
    args.output_dir = os.path.join(parent, f'{folder}/crossattention_map_check')

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

    print(f" (1.3) save dir")
    output_dir = args.output_dir
    parent, network_dir = os.path.split(args.network_weights)
    model_name = os.path.splitext(network_dir)[0]
    if 'last' not in model_name:
        model_epoch = int(model_name.split('-')[-1])
    else:
        model_epoch = 'last'

    output_dir = os.path.join(output_dir, f'lora_{model_epoch}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'final output dir : {output_dir}')

    print(f' \n step 2. make stable diffusion model')
    device = accelerator.device

    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")



    print(f' \n step 3. make lora model')
    sys.path.append(os.path.dirname(__file__))
    network_module = importlib.import_module(args.network_module)
    net_kwargs = {}
    network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet,
                                            neuron_dropout=args.network_dropout, **net_kwargs, )
    print(f' (2.5.3) apply trained state dict')
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
    network.to(device)

    print(f' (2.4.+) model to accelerator device')
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    network.to(accelerator.device, dtype=weight_dtype)

    controller = AttentionStore()
    register_attention_control(unet, controller)

    print(f' \n step 3. ground-truth image preparing')
    print(f' (3.1) prompt condition')
    prompt = 'crack'
    context = init_prompt(tokenizer, text_encoder, device, prompt)
    uncon, con = torch.chunk(context, 2)

    print(f' (3.2) train images')
    trg_h, trg_w = args.resolution

    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    print(f' (3.3) test images')
    ddim_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    ddim_scheduler.set_timesteps(50)
    for t in ddim_scheduler.timesteps:
        # 1. predict noise model_output
        model_output = unet(image, t, con).sample
        image = ddim_scheduler(model_output, t, image, eta=0.0,
                               use_clipped_model_output=None,).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = numpy_to_pil(image)




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
    parser.add_argument("--final_noising_time", type=int, default = 250)
    parser.add_argument("--mask_thredhold", type=float, default = 0.5)
    parser.add_argument("--pixel_mask_res", type=float, default=0.1)
    parser.add_argument("--pixel_thred", type=float, default=0.1)
    parser.add_argument("--inner_iteration", type=int, default=10)
    parser.add_argument("--org_latent_attn_map_check", action = 'store_true')
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
    parser.add_argument("--cross_map_res", type=arg_as_list, default=[64,32,16,8])
    args = parser.parse_args()
    main(args)