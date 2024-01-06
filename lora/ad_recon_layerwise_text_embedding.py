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
from utils.image_utils import latent2image
from utils.scheduling_utils import prev_step
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None


def register_attention_control(unet: nn.Module, controller: AttentionStore,
                               mask_threshold: float = 1):  # if mask_threshold is 1, use itself

    def ca_forward(self, layer_name):
        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True

                if trg_indexs_list is not None:

                    if 'mid' in layer_name:
                        trg_num = 6
                    else:
                        if 'down' in layer_name:
                            position_num = 0
                            if 'blocks_0' in layer_name:
                                block_num = 0
                            elif 'blocks_1' in layer_name:
                                block_num = 2
                            elif 'blocks_2' in layer_name:
                                block_num = 4

                            if 'attentions_0' in layer_name:
                                attention_num = 0
                            elif 'attentions_1' in layer_name:
                                attention_num = 1
                        elif 'up' in layer_name:
                            position_num = 7
                            if 'blocks_1' in layer_name:
                                block_num = 0
                            elif 'blocks_2' in layer_name:
                                block_num = 3
                            elif 'blocks_3' in layer_name:
                                block_num = 6

                            if 'attentions_0' in layer_name:
                                attention_num = 0
                            elif 'attentions_1' in layer_name:
                                attention_num = 1
                            elif 'attentions_2' in layer_name:
                                attention_num = 2
                        trg_num = position_num + block_num + attention_num
                    trg_num = trg_num + 1
                    cls_emb = context[:, 0, :]
                    other_emb = context[:, 17:, :]
                    trgger_emb = context[:, trg_num, :]
                    if cls_emb.dim() != 3:
                        cls_emb = cls_emb.unsqueeze(0)
                    if cls_emb.dim() != 3:
                        cls_emb = cls_emb.unsqueeze(0)
                    if trgger_emb.dim() != 3:
                        trgger_emb = trgger_emb.unsqueeze(0)
                    if trgger_emb.dim() != 3:
                        trgger_emb = trgger_emb.unsqueeze(0)
                    context = torch.cat([cls_emb, trgger_emb, other_emb], dim=1)

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
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                            device=query.device), query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)
            if is_cross_attention:
                controller.store(attention_probs[:,:,:3], layer_name)

            if is_cross_attention and mask is not None:
                if layer_name in mask.keys() :
                    mask = mask[layer_name].unsqueeze(-1)
                    mask = mask.repeat(1, 1, attention_probs.shape[-1]).to(attention_probs.device) # head, pix_num, sen_len
                    #print(mask.shape)
                    z_attn_probs, x_attn_probs = attention_probs.chunk(2, dim=0) # head, pix_num, sen_len
                    x_attn_probs = z_attn_probs * mask + x_attn_probs * (1 - mask)
                    attention_probs = torch.cat([z_attn_probs, x_attn_probs], dim=0)
                #else :
                #    z_attn_probs, x_attn_probs = attention_probs.chunk(2, dim=0)  # head, pix_num, sen_len
                #    x_attn_probs = z_attn_probs
                #    attention_probs = torch.cat([z_attn_probs, x_attn_probs], dim=0)


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

    parent = os.path.split(args.network_weights)[0] # unique_folder,
    args.output_dir = os.path.join(parent, f'recon_infer_without_thredhold/ddim_step_{args.num_ddim_steps}_cross_map_res_{args.cross_map_res[0]}_unet_thred_{args.mask_thredhold}')
    os.makedirs(args.output_dir, exist_ok=True)

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
    parent = os.path.split(args.network_weights)[0] # unique_folder,
    text_embedding_base_dir = os.path.join(parent, 'text_embedding')
    network_weights = os.listdir(args.network_weights)

    for weight in network_weights:
        weight_dir = os.path.join(args.network_weights, weight)
        parent, network_dir = os.path.split(weight_dir)
        model_name = os.path.splitext(network_dir)[0]
        if 'last' not in model_name:
            model_epoch = int(model_name.split('-')[-1])
            lora_epoch = str(model_name.split('-')[-1])
        else:
            model_epoch = 'last'
            lora_epoch = 'last'
        text_embedding_dir = os.path.join(text_embedding_base_dir, f'training_text_embeddings-{lora_epoch}.pt')

        flag = False
        if args.recon_only_trg_epoch :
            if args.trg_epoch in weight :
                flag = True
        else :
            flag = True

        if flag == True :

            if args.detail_64:
                save_dir = os.path.join(output_dir, f'lora_epoch_{model_epoch}_res_64_change_more_lora')
            else :
                save_dir = os.path.join(output_dir, f'lora_epoch_{model_epoch}')
            os.makedirs(save_dir, exist_ok=True)

            print(f' \n step 2. make stable diffusion model')
            device = accelerator.device
            print(f' (2.1) tokenizer')
            tokenizer = train_util.load_tokenizer(args)
            print(f' (2.2) SD')
            invers_text_encoder, vae, invers_unet, load_stable_diffusion_format = train_util._load_target_model(args,
                                                                                                                weight_dtype,
                                                                                                                device,
                                                                                                                unet_use_linear_projection_in_v2=False, )
            invers_text_encoders = invers_text_encoder if isinstance(invers_text_encoder, list) else [invers_text_encoder]
            text_encoder, vae, unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype, device,
                                                                                                  unet_use_linear_projection_in_v2=False, )
            text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
            vae.to(accelerator.device, dtype=vae_dtype)

            print(f' (2.4) scheduler')
            scheduler_cls = get_scheduler(args.sample_sampler, args.v_parameterization)[0]
            scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps, beta_start=args.scheduler_linear_start,
                                      beta_end=args.scheduler_linear_end, beta_schedule=args.scheduler_schedule)
            scheduler.set_timesteps(args.num_ddim_steps)
            inference_times = scheduler.timesteps

            print(f' (2.4.+) model to accelerator device')
            if len(invers_text_encoders) > 1:
                invers_unet, invers_t_enc1, invers_t_enc2 = invers_unet.to(device), invers_text_encoders[0].to(device), \
                invers_text_encoders[1].to(device)
                invers_text_encoder = [invers_t_enc1, invers_t_enc2]
                del invers_t_enc1, invers_t_enc2
                unet, t_enc1, t_enc2 = unet.to(device), text_encoders[0].to(device), text_encoders[1].to(device)
                text_encoder = [t_enc1, t_enc2]
                del t_enc1, t_enc2
            else:
                invers_unet, invers_text_encoder = invers_unet.to(device), invers_text_encoder.to(device)
                unet, text_encoder = unet.to(device), text_encoder.to(device)

            print(f' (2.5) network')
            sys.path.append(os.path.dirname(__file__))
            network_module = importlib.import_module(args.network_module)
            print(f' (2.5.1) merging weights')
            net_kwargs = {}
            if args.network_args is not None:
                for net_arg in args.network_args:
                    key, value = net_arg.split("=")
                    net_kwargs[key] = value
            print(f' (2.5.2) make network')
            sys.path.append(os.path.dirname(__file__))
            network_module = importlib.import_module(args.network_module)
            net_kwargs = {}
            network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet,
                                                    neuron_dropout=args.network_dropout, **net_kwargs, )
            print(f' (2.5.3) apply trained state dict')
            network.apply_to(text_encoder, unet, apply_text_encoder=False, apply_unet=True)
            if args.network_weights is not None:
                info = network.load_weights(weight_dir)
            network.to(device)
            text_embedding = torch.load(text_embedding_dir, map_location=torch.device('cpu'))
            text_embedding = text_embedding.to(device, weight_dtype)

            print(f' (2.4.+) model to accelerator device')
            controller = AttentionStore()
            register_attention_control(unet, controller)

            print(f' \n step 3. ground-truth image preparing')
            print(f' (3.1) prompt condition')
            context = init_prompt(tokenizer, text_encoder, device, args.prompt)
            uncon, con = torch.chunk(context, 2)
            uncon, con = uncon[:, :3,:], con[:, :3,:]

            print(f' (3.2) train images')
            trg_h, trg_w = args.resolution

            print(f' (3.3) test images')
            test_img_folder = os.path.join(args.concept_image_folder, 'test_ex/bad')
            test_mask_folder = os.path.join(args.concept_image_folder, 'test_ex/corrected')
            classes = os.listdir(test_img_folder)

            for class_name in classes:
                if '_' in class_name:
                    trg_prompt = class_name.split('_')[-1]
                else:
                    trg_prompt = class_name
                if 'contamination' in class_name :

                    class_base_folder = os.path.join(save_dir, class_name)
                    os.makedirs(class_base_folder, exist_ok=True)

                    image_folder = os.path.join(test_img_folder, class_name)
                    mask_folder = os.path.join(test_mask_folder, class_name)
                    invers_context = init_prompt(tokenizer, invers_text_encoder, device, f'a photo of {trg_prompt}')
                    inv_unc, inv_c = invers_context.chunk(2)
                    test_images = os.listdir(image_folder)

                    for j, test_image in enumerate(test_images):
                        if '001' in test_image:

                            name, ext = os.path.splitext(test_image)
                            trg_img_output_dir = os.path.join(class_base_folder, f'{name}')
                            os.makedirs(trg_img_output_dir, exist_ok=True)

                            test_img_dir = os.path.join(image_folder, test_image)
                            shutil.copy(test_img_dir, os.path.join(trg_img_output_dir, test_image))

                            mask_img_dir = os.path.join(mask_folder, test_image)
                            shutil.copy(mask_img_dir, os.path.join(trg_img_output_dir, f'{name}_mask{ext}'))

                            print(f' (2.3.1) inversion')
                            with torch.no_grad():
                                org_img = load_image(test_img_dir, 512, 512)
                                org_vae_latent = image2latent(org_img, vae, device, weight_dtype)
                            # ------------------------------ generate attn mask map ------------------------------ #
                            with torch.no_grad():
                                inf_time = inference_times.tolist()
                                inf_time.reverse()  # [0,20,40,60,80,100 , ... 980]
                                back_dict = {}
                                latent = org_vae_latent
                                with torch.no_grad():
                                    mask_dict = {}
                                    mask_dict_avg = {}
                                    for i, t in enumerate(inf_time[:-1]):
                                        if i == 0 :
                                            next_time = inf_time[i + 1]
                                            if next_time <= args.final_noising_time:
                                                back_dict[int(t)] = latent

                                                from utils.model_utils import call_unet

                                                cls_embedding = con[:, 0, :]
                                                if cls_embedding.dim() != 3:
                                                    cls_embedding = cls_embedding.unsqueeze(0)
                                                if cls_embedding.dim() != 3:
                                                    cls_embedding = cls_embedding.unsqueeze(0)
                                                other_embedding = con[:, 2:, :]
                                                if other_embedding.dim() != 3:
                                                    other_embedding = other_embedding.unsqueeze(0)
                                                embedding = torch.cat((cls_embedding, text_embedding, other_embedding),dim=1)
                                                noise_pred = call_unet(unet, latent, t, embedding, [[1]], None)
                                                attn_stores = controller.step_store
                                                controller.reset()
                                                mask_dict_avg_sub = {}

                                                for layer_name in attn_stores:
                                                    attn = attn_stores[layer_name][0].squeeze()  # head, pix_num
                                                    res = int(attn.shape[1] ** 0.5)
                                                    if res in args.cross_map_res:

                                                        if 'down' in layer_name:
                                                            key_name = f'down_{res}'
                                                        elif 'up' in layer_name:
                                                            key_name = f'up_{res}'
                                                        else:
                                                            key_name = f'mid_{res}'
                                                        if key_name not in mask_dict_avg_sub :
                                                            mask_dict_avg_sub[key_name] = []
                                                        mask_dict_avg_sub[key_name].append(attn)

                                                        cls_score, trigger_score, pad_score = attn.chunk(3, dim=-1)
                                                        h = cls_score.shape[0]
                                                        trigger_score = trigger_score.unsqueeze(-1)        # head, pix_num, 1
                                                        trigger_score = trigger_score.reshape(h, res, res) # head, res, res
                                                        trigger_score = trigger_score.mean(dim=0) # res, res
                                                        trigger = trigger_score / trigger_score.max()      # res, res
                                                        pil_img = Image.fromarray((trigger.cpu().numpy() * 255).astype(np.uint8))
                                                        pil_img = pil_img.resize((512, 512), Image.BILINEAR)
                                                        pil_img.save(os.path.join(trg_img_output_dir, f'unet_mask_{layer_name}.png'))
                                                        anormal_map = torch.flatten(trigger).unsqueeze(0)  # 1, res*res
                                                        mask_dict[layer_name] = anormal_map
                                                if args.detail_64 :
                                                    for mask_dict_key in mask_dict.keys():
                                                        if 'down_blocks_0' in mask_dict_key:
                                                            sub_key_name = mask_dict_key.replace('down_blocks_0', 'up_blocks_3')
                                                            mask_dict[mask_dict_key] = mask_dict[sub_key_name]



                                                # attn_dict
                                                for key_name in mask_dict_avg_sub:
                                                    attn_list = mask_dict_avg_sub[key_name]
                                                    attn = torch.cat(attn_list, dim=0)
                                                    cls_score, trigger_score, pad_score = attn.chunk(3, dim=-1) # head, pix_num
                                                    res = int(trigger_score.shape[1] ** 0.5)
                                                    h = trigger_score.shape[0]
                                                    trigger_score = trigger_score.unsqueeze(-1)        # head, pix_num, 1
                                                    trigger_score = trigger_score.reshape( h, res, res) # head, res, res
                                                    trigger_score = trigger_score.mean(dim=0)           # res, res
                                                    trigger = trigger_score / trigger_score.max()
                                                    mask_dict_avg[key_name] = trigger                   # up_64

                            # ------------------------------ generate background latent ------------------------------ #
                            from utils.scheduling_utils import next_step

                            with torch.no_grad() :
                                time_steps = []
                                with torch.no_grad():
                                    for i, t in enumerate(inf_time):
                                        time_steps.append(t)
                                        back_dict[int(t)] = latent
                                        time_steps.append(t)
                                        noise_pred = call_unet(unet, latent, t, inv_c, None, None)
                                        latent = next_step(noise_pred, int(t), latent, scheduler)
                            time_steps.reverse()

                            # ------------------------------ generate new latent ------------------------------ #
                            x_latent_dict = {}
                            x_latent_dict[t] = back_dict[t]
                            with torch.no_grad():
                                for j, t in enumerate(time_steps[:-1]):
                                    prev_time = time_steps[j + 1]
                                    z_latent = back_dict[prev_time]
                                    x_latent = x_latent_dict[t]
                                    input_latent = torch.cat([z_latent, x_latent], dim=0)
                                    #
                                    cls_embedding = con[:, 0, :]
                                    if cls_embedding.dim() != 3:
                                        cls_embedding = cls_embedding.unsqueeze(0)
                                    if cls_embedding.dim() != 3:
                                        cls_embedding = cls_embedding.unsqueeze(0)
                                    other_embedding = con[:, 2:, :]
                                    if other_embedding.dim() != 3:
                                        other_embedding = other_embedding.unsqueeze(0)
                                    embedding = torch.cat((cls_embedding, text_embedding, other_embedding), dim=1)

                                    input_cont = torch.cat([embedding, embedding], dim=0)
                                    noise_pred = call_unet(unet, input_latent, t, input_cont, None, mask_dict)
                                    controller.reset()
                                    z_noise_pred, x_noise_pred = noise_pred.chunk(2, dim=0)
                                    x_latent = prev_step(x_noise_pred, int(t), x_latent, scheduler)
                                    # ------------------------------ pixel recon ------------------------------ #
                                    if args.pixel_copy :
                                        print(f'keys in mask_dict_avg : {mask_dict_avg.keys()}')
                                        for key in mask_dict_avg.keys():
                                            up_key = False
                                            if 'up' in key :
                                                pixel_mask = mask_dict_avg[key].to(z_latent.device)
                                                # ----------------------------------------------------------------------
                                                pixel_save_mask_np = pixel_mask.cpu().numpy()
                                                pixel_mask_img = (pixel_save_mask_np * 255).astype(np.uint8)
                                                pil_img_512 = Image.fromarray(pixel_mask_img).resize((512, 512))

                                                pil_img_64 = Image.fromarray(pixel_mask_img).resize((64, 64))
                                                from torchvision import transforms
                                                pixel_mask =  transforms.ToTensor()(pil_img_64)
                                                if prev_time == 0 :
                                                    pil_img_512.save(os.path.join(trg_img_output_dir, f'{name}_pixel_mask{ext}'))
                                                    up_key = True
                                            if up_key :
                                                # ----------------------------------------------------------------------
                                                pixel_mask = pixel_mask.unsqueeze(0).unsqueeze(0) # 1, 1, res, res
                                                pixel_mask = pixel_mask.repeat(1, 4, 1, 1) # 1, 4, res, res
                                                x_latent = z_latent * pixel_mask + x_latent * (1 - pixel_mask)
                                                # ------------------------------- latent2img ------------------------------- #
                                    pil_img = Image.fromarray(latent2image(x_latent, vae, return_type='np'))
                                    pil_img.save(os.path.join(trg_img_output_dir, f'{name}_recon_{prev_time}{ext}'))
                                    x_latent_dict[prev_time] = x_latent

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
    parser.add_argument("--pixel_thredhold", type=float, default=0.1)
    parser.add_argument("--pixel_copy", action = 'store_true')
    parser.add_argument('--train_down', nargs='+', type=int, help='use which res layers in U-Net down', default=[])
    parser.add_argument('--train_mid', nargs='+', type=int, help='use which res layers in U-Net mid', default=[8])
    parser.add_argument('--train_up', nargs='+', type=int, help='use which res layers in U-Net up', default=[16,32,64])
    parser.add_argument("--detail_64", action='store_true')
    parser.add_argument("--recon_only_trg_epoch", action='store_true')
    parser.add_argument("--trg_epoch", type=str)


    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--cross_map_res", type=arg_as_list, default=[64,32,16,8])
    args = parser.parse_args()
    main(args)