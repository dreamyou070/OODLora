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
from utils.model_utils import call_unet
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
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

        def forward(hidden_states, context=None, attention_mask=None, temb=None, ):
            is_cross = context is not None

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (hidden_states.shape if context is None else context.shape)
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if context is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(context)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)

            attention_probs = controller(attention_probs, is_cross, place_in_unet)

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


def register_attention_control(unet: nn.Module, controller: AttentionStore,  mask_thredhold: float = 1):  # if mask_threshold is 1, use itself

    map_dict = {}

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
                            device=query.device), query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            if is_cross_attention and trg_indexs_list is not None:

                if attention_scores.shape[0] == 8 :
                    cls_map = attention_probs[:, :, 0].unsqueeze(-1)
                    hole_map = attention_probs[:, :, 1].unsqueeze(-1)
                    crack_map = attention_probs[:, :, 2].unsqueeze(-1)  # head, pixel_num, 1

                    maps = torch.cat([cls_map, hole_map, crack_map], dim=-1)
                    controller.store(maps, layer_name)

                else :
                    if layer_name in mask.keys() :
                        position_map = mask[layer_name]
                        print(f'type of positionmap in infer : {type(position_map)}')
                        background_attention_probs, object_attention_probs = attention_probs.chunk(2, dim=0)

                        batch_num = len(trg_indexs_list)

                        attention_probs_back_batch = torch.chunk(background_attention_probs, batch_num, dim=0)
                        attention_probs_object_batch = torch.chunk(object_attention_probs, batch_num, dim=0) #  torch.Size([8, 4096, 77])

                        for batch_idx, (attention_probs_back, attention_probs_object) in enumerate(zip(attention_probs_back_batch,attention_probs_object_batch)):

                            pixel_num = attention_probs_back.shape[1] # head, pixel_num, word_num
                            #map_list = []
                            res = int(pixel_num ** 0.5)
                            if res in args.cross_map_res :
                                query = self.to_q(hidden_states)
                                query = self.reshape_heads_to_batch_dim(query)
                                back_query, object_query = query.chunk(2, dim=0)
                                #map_list.append(position_map)
                                position_map = position_map.unsqueeze(-1) # head, pixel_num, 1
                                position_map = position_map.expand(object_query.shape)
                                object_query = object_query * (1-position_map) + back_query * (position_map)
                                query = torch.cat([back_query, object_query], dim=0)
                                attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                                                                             device=query.device), query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
                                attention_probs = attention_scores.softmax(dim=-1)


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

def main(args) :

    parent = os.path.split(args.network_weights)[0]
    folder = os.path.split(parent)[-1]
    args.output_dir = os.path.join(parent, f'{folder}/pixel_crossattention_bland_inference')

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

    print(f' \n step 2. make stable diffusion model')
    device = accelerator.device

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    inverse_text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    inverse_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

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

    """
    trg_resolutions = args.cross_map_res
    title = ''
    for res in trg_resolutions:
        title += f'_{res}'
    print(f'title : {title}')

    output_dir = os.path.join(output_dir,f'lora_{model_epoch}_final_noising_{args.final_noising_time}_num_ddim_steps_{args.num_ddim_steps}_'
                                         f'cross_res{title}_'
                                         f'res_{args.pixel_mask_res}_'
                                         f'pixel_mask_pixel_thred_{args.pixel_thred}_'
                                         f'inner_iter_{args.inner_iteration}_'
                                         f'mask_thredhold_{args.mask_thredhold}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'final output dir : {output_dir}')

    print(f' (2.4) scheduler')
    scheduler_cls = get_scheduler(args.sample_sampler, args.v_parameterization)[0]
    scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps, beta_start=args.scheduler_linear_start,
                              beta_end=args.scheduler_linear_end, beta_schedule=args.scheduler_schedule)
    scheduler.set_timesteps(args.num_ddim_steps)
    inference_times = scheduler.timesteps

    print(f' (2.4.+) model to accelerator device')
    
    
    controller = AttentionStore()
    register_attention_control(unet, controller, mask_thredhold=args.mask_thredhold)

    print(f' \n step 3. ground-truth image preparing')
    print(f' (3.1) prompt condition')
    prompt = 'hole crack'
    context = init_prompt(tokenizer, text_encoder, device, prompt)
    uncon, con = torch.chunk(context, 2)

    print(f' (3.2) train images')
    trg_h, trg_w = args.resolution

    print(f' (3.3) test images')
    test_img_folder = os.path.join(args.concept_image_folder, 'test_ex/bad')
    test_mask_folder = os.path.join(args.concept_image_folder, 'test_ex/corrected')
    classes = os.listdir(test_img_folder)

    for class_name in classes:
        if 'bad' not in class_name:
            class_base_folder = os.path.join(output_dir, class_name)
            os.makedirs(class_base_folder, exist_ok=True)

            image_folder = os.path.join(test_img_folder, class_name)
            mask_folder = os.path.join(test_mask_folder, class_name)

            invers_context = init_prompt(tokenizer, invers_text_encoder, device, f'a photo of {class_name}')
            inv_unc, inv_c = invers_context.chunk(2)
            test_images = os.listdir(image_folder)

            for j, test_image in enumerate(test_images):

                name, ext = os.path.splitext(test_image)
                trg_img_output_dir = os.path.join(class_base_folder, f'{name}')
                os.makedirs(trg_img_output_dir, exist_ok=True)

                test_img_dir = os.path.join(image_folder, test_image)
                shutil.copy(test_img_dir, os.path.join(trg_img_output_dir, test_image))

                mask_img_dir = os.path.join(mask_folder, test_image)
                shutil.copy(mask_img_dir, os.path.join(trg_img_output_dir, f'{name}_mask{ext}'))
                mask_np = load_image(mask_img_dir, trg_h=int(trg_h), trg_w=int(trg_w))
                mask_np = np.where(mask_np > 100, 1, 0)  # binary mask
                gt_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)

                print(f' (2.3.1) inversion')
                image_gt_np = load_image(test_img_dir, trg_h=int(trg_h), trg_w=int(trg_w))

                with torch.no_grad():
                    org_vae_latent  = image2latent(image_gt_np, vae, device=device, weight_dtype=weight_dtype)

                if args.org_latent_attn_map_check :
                    input_latent = org_vae_latent
                    input_context = con
                    noise_pred = call_unet(unet, input_latent, 0, input_context, [[1]], None)
                    attn_store_dict = controller.step_store
                    controller.reset()
                    for layer in attn_store_dict.keys():
                        attn_map = attn_store_dict[layer][0]
                        cls_map, hole_map, crack_map = torch.chunk(attn_map, 3, dim=-1)
                        cls_map, hole_map, crack_map = cls_map.squeeze(), hole_map.squeeze(), crack_map.squeeze() # head, res*res
                        res = int(cls_map.shape[1] ** 0.5)
                        print(f'layer : {layer}, trigger word map shape : {attn_map.shape}')

                        cls_map = (torch.sum(cls_map, dim=0) / cls_map.shape[0]).unsqueeze(0).reshape(res, res)
                        cls_pil = Image.fromarray((np.array(cls_map.cpu().detach()) * 255).astype(np.uint8)).resize((512,512))
                        cls_pil.save(os.path.join(trg_img_output_dir, f'cls_{class_name}_{name}_{layer}_attn_map.png'))

                        hole_map = (torch.sum(hole_map, dim=0) / hole_map.shape[0]).unsqueeze(0).reshape(res, res)
                        hole_pil = Image.fromarray((np.array(hole_map.cpu().detach()) * 255).astype(np.uint8)).resize((512, 512))
                        hole_pil.save(os.path.join(trg_img_output_dir, f'hole_{class_name}_{name}_{layer}_attn_map.png'))

                        crack_map = (torch.sum(crack_map, dim=0) / crack_map.shape[0]).unsqueeze(0).reshape(res, res)
                        crack_pil = Image.fromarray((np.array(crack_map.cpu().detach()) * 255).astype(np.uint8)).resize((512, 512))
                        crack_pil.save(os.path.join(trg_img_output_dir, f'crack_{class_name}_{name}_{layer}_attn_map.png'))

                        total_map = (cls_map + hole_map + crack_map)
                        print(f'total_map : {total_map}')

                        # head==8, pix_num, 1




                else :
                    with torch.no_grad():
                        inf_time = inference_times.tolist()
                        inf_time.reverse()  # [0,20,40,60,80,100 , ... 980]
                        print(f'inf_time : {inf_time}')
                        org_latent_dict, time_steps, pil_images = ddim_loop(args,
                                                                            latent=org_vae_latent,
                                                                            context=inv_c,
                                                                            inference_times=inf_time,
                                                                            scheduler=scheduler,
                                                                            unet=invers_unet,
                                                                            vae=vae,
                                                                            final_time=args.final_noising_time,
                                                                            base_folder_dir=trg_img_output_dir,
                                                                            name=name)
                        noising_times = org_latent_dict.keys()
                        print(f'noiseing_times : {noising_times}')
                        st_noise_latent = org_latent_dict[args.final_noising_time]

                        time_steps.reverse()
                        
                        recon_loop(args,
                                   org_latent_dict,
                                   start_latent=st_noise_latent,
                                   gt_pil = gt_pil,
                                   context=context,
                                   inference_times= time_steps,
                                   scheduler=scheduler,
                                   unet=unet,
                                   vae=vae,
                                   base_folder_dir=trg_img_output_dir,
                                   controller=controller,
                                   name=name,weight_dtype=weight_dtype)
                        
                break
    """


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
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--cross_map_res", type=arg_as_list, default=[64,32,16,8])
    args = parser.parse_args()
    main(args)