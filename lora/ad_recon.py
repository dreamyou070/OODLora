import argparse
from accelerate.utils import set_seed
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
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
from attention_store import AttentionStore
import torch.nn as nn
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None
from utils.model_utils import call_unet
from utils.scheduling_utils import next_step

def register_attention_control(unet: nn.Module, controller: AttentionStore,
                               mask_threshold: float = 1):  # if mask_threshold is 1, use itself

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
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                            device=query.device), query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            if is_cross_attention:
                controller.store(attention_probs[:,:,:args.truncate_length], layer_name)

            if is_cross_attention and mask is not None:
                if layer_name in mask.keys() :
                    mask = mask[layer_name].unsqueeze(-1)
                    mask = mask.repeat(1, 1, attention_probs.shape[-1]).to(attention_probs.device) # head, pix_num, sen_len
                    z_attn_probs, x_attn_probs = attention_probs.chunk(2, dim=0) # head, pix_num, sen_len
                    x_attn_probs = z_attn_probs * mask + x_attn_probs * (1 - mask)
                    attention_probs = torch.cat([z_attn_probs, x_attn_probs], dim=0)



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
    args.output_dir = os.path.join(parent, f'recon_infer/step_{args.num_ddim_steps}_'
                                           f'guidance_scale_{args.guidance_scale}_'
                                           f'start_from_origin_{args.start_from_origin}_'
                                           f'start_from_final_{args.start_from_final}_'
                                           f'inner_iteration_{args.inner_iteration}')

    print(f'saving will be on {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f' \n step 1. setting')
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    if args.seed is None: args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f'\n step 2. preparing accelerator')
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    print(f" (2.1) save dir")

    def get_latent_mask(normalized_mask, trg_res):
        pixel_save_mask_np = normalized_mask.cpu().numpy()
        pixel_mask_img = (pixel_save_mask_np * 255).astype(np.uint8)
        latent_mask_pil = Image.fromarray(pixel_mask_img).resize(
            (trg_res, trg_res,))
        latent_mask_np = np.array(latent_mask_pil)
        latent_mask_np = latent_mask_np / latent_mask_np.max()  # 64,64
        latent_mask_torch = torch.from_numpy(latent_mask_np).to(latent.device,
                                                                dtype=weight_dtype)
        latent_mask_torch = latent_mask_torch.unsqueeze(0).unsqueeze(0)
        latent_mask = latent_mask_torch.repeat(1, 4, 1, 1)
        return latent_mask_np, latent_mask

    output_dir = args.output_dir
    network_weights = os.listdir(args.network_weights)
    for weight in network_weights:
        weight_dir = os.path.join(args.network_weights, weight)
        if args.trg_lora_epoch in weight:
            parent, network_dir = os.path.split(weight_dir)
            model_name = os.path.splitext(network_dir)[0]
            if 'last' not in model_name:
                model_epoch = int(model_name.split('-')[-1])
            else:
                model_epoch = 'last'
            save_dir = os.path.join(output_dir, f'lora_epoch_{model_epoch}')
            os.makedirs(save_dir, exist_ok=True)

            print(f' \n step 2. make stable diffusion model')
            device = accelerator.device
            print(f' (2.1) tokenizer')
            tokenizer = train_util.load_tokenizer(args)
            print(f' (2.2) SD')
            text_encoder, vae, unet, load_stable_diffusion_format = train_util._load_target_model(args, weight_dtype, device,
                                                                                                  unet_use_linear_projection_in_v2=False, )
            vae.to(accelerator.device, dtype=vae_dtype)

            print(f' (2.4) scheduler')
            scheduler_cls = get_scheduler(args.sample_sampler, args.v_parameterization)[0]
            scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps, beta_start=args.scheduler_linear_start,
                                      beta_end=args.scheduler_linear_end, beta_schedule=args.scheduler_schedule)
            scheduler.set_timesteps(args.num_ddim_steps)
            inference_times = scheduler.timesteps

            print(f' (2.4.+) model to accelerator device')
            unet, text_encoder = unet.to(device), text_encoder.to(device)

            print(f' (2.5) network')
            sys.path.append(os.path.dirname(__file__))
            network_module = importlib.import_module(args.network_module)
            net_kwargs = {}
            network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet,
                                                    neuron_dropout=args.network_dropout, **net_kwargs, )
            print(f' (2.5.3) apply trained state dict')
            network.apply_to(text_encoder, unet, True, True)
            if args.network_weights is not None:
                info = network.load_weights(weight_dir)
            network.to(device)

            print(f' (2.4.+) model to accelerator device')
            controller = AttentionStore()
            register_attention_control(unet, controller)

            print(f' \n step 3. ground-truth image preparing')
            print(f' (3.1) prompt condition')
            context = init_prompt(tokenizer, text_encoder, device, args.prompt,
                                  args.negative_prompt)
            uncon, con = torch.chunk(context, 2)
            uncon, con = uncon[:, :,:], con[:, :,:]

            print(f' (3.2) test images')
            test_img_folder = os.path.join(args.concept_image_folder, 'test_ex/rgb')
            test_mask_folder = os.path.join(args.concept_image_folder, 'test_ex/gt')
            classes = os.listdir(test_img_folder)

            for class_name in classes:

                class_base_folder = os.path.join(save_dir, class_name)
                os.makedirs(class_base_folder, exist_ok=True)

                image_folder = os.path.join(test_img_folder, class_name)
                mask_folder = os.path.join(test_mask_folder, class_name)

                test_images = os.listdir(image_folder)

                for j, test_image in enumerate(test_images):

                    name, ext = os.path.splitext(test_image)
                    trg_img_output_dir = os.path.join(class_base_folder, f'{name}')
                    os.makedirs(trg_img_output_dir, exist_ok=True)

                    test_img_dir = os.path.join(image_folder, test_image)
                    Image.open(test_img_dir).convert('RGB').resize((512,512)).save(os.path.join(trg_img_output_dir,f'{name}_org{ext}'))

                    mask_img_dir = os.path.join(mask_folder, test_image)
                    Image.open(mask_img_dir).convert('L').resize((512, 512)).save(os.path.join(trg_img_output_dir, f'{name}_mask{ext}'))

                    print(f' (2.3.1) inversion')
                    with torch.no_grad():

                        pipeline = StableDiffusionLongPromptWeightingPipeline(vae=vae,
                                                                              text_encoder=text_encoder,
                                                                              tokenizer=tokenizer,
                                                                              unet=unet,
                                                                              scheduler=scheduler,
                                                                              safety_checker=None,
                                                                              feature_extractor=None,
                                                                              requires_safety_checker=False, )

                        org_img = load_image(test_img_dir, 512, 512)
                        org_vae_latent = image2latent(org_img, vae, device, weight_dtype)

                        call_unet(unet, org_vae_latent, 0, con[:, :args.truncate_length, :], None, None)


                        # -------------------------------------------- [1] generate attn mask map ---------------------------------------------- #
                        inf_time = inference_times.tolist()
                        inf_time.reverse()  # [0,250,500,750]
                        back_dict = {}
                        latent = org_vae_latent
                        mask_dict_avg = {}
                        back_dict[0] = latent
                        attn_stores = controller.step_store
                        controller.reset()
                        mask_dict_avg_sub = {}
                        for layer_name in attn_stores:
                            attn = attn_stores[layer_name][0].squeeze()  # head, pix_num
                            res = int(attn.shape[1] ** 0.5)
                            if res in args.cross_map_res:
                                if 'down' in layer_name:
                                    key_name = f'down_{res}'
                                    pos = 'down'
                                elif 'up' in layer_name:
                                    key_name = f'up_{res}'
                                    pos = 'up'
                                else:
                                    key_name = f'mid_{res}'
                                    pos = 'mid'
                                if 'attentions_0' in layer_name:
                                    part = 'attn_0'
                                elif 'attentions_1' in layer_name:
                                    part = 'attn_1'
                                else:
                                    part = 'attn_2'

                                if args.use_avg_mask:
                                    if key_name not in mask_dict_avg_sub:
                                        mask_dict_avg_sub[key_name] = []
                                    mask_dict_avg_sub[key_name].append(attn)
                                else :
                                    if res in args.cross_map_res and pos in args.trg_position and part == args.trg_part:
                                        if args.truncate_length == 3 :
                                            cls_score, trigger_score, pad_score = attn.chunk(3, dim=-1)  # head, pix_num
                                        else :
                                            cls_score, trigger_score = attn.chunk(2, dim=-1)  # head, pix_num

                                        h = trigger_score.shape[0]
                                        trigger_score = trigger_score.unsqueeze(-1).reshape(h, res, res)
                                        trigger_score = trigger_score.mean(dim=0)  # res, res
                                        pixel_mask = trigger_score / trigger_score.max()  # res, res
                                        # ------------------------------------------------------------------------------------------------------------------------

                                        latent_mask_np, latent_mask = get_latent_mask(pixel_mask, 64)
                                        Image.fromarray((latent_mask_np * 255).astype(np.uint8)).resize((512, 512)).save(
                                            os.path.join(trg_img_output_dir, f'{name}_pixel_mask{ext}'))
                        if args.use_avg_mask:
                            for key_name in mask_dict_avg_sub:
                                attn_list = mask_dict_avg_sub[key_name]
                                attn = torch.cat(attn_list, dim=0)
                                if args.truncate_length == 3:
                                    cls_score, trigger_score, pad_score = attn.chunk(3, dim=-1)  # head, pix_num
                                else:
                                    cls_score, trigger_score = attn.chunk(2, dim=-1)  # head, pix_num
                                res = int(trigger_score.shape[1] ** 0.5)
                                h = trigger_score.shape[0]
                                trigger_score = trigger_score.unsqueeze(-1)  # head, pix_num, 1
                                trigger_score = trigger_score.reshape(h, res, res)  # head, res, res
                                trigger_score = trigger_score.mean(dim=0)  # res, res
                                mask_dict_avg[key_name] = trigger_score / trigger_score.max()

                            for key in mask_dict_avg.keys():
                                pixel_mask = mask_dict_avg[key].to(latent.device)
                                latent_mask_np, latent_mask = get_latent_mask(pixel_mask, 64)
                                Image.fromarray((latent_mask_np * 255).astype(np.uint8)).resize((512, 512)).save(
                                    os.path.join(trg_img_output_dir, f'{name}_pixel_mask{ext}'))


                        # -------------------------------------------- [2] generate background latent ---------------------------------------------- #
                        time_steps = []
                        inf_time.append(999) # 0, 250, 500, 750, 999
                        for i, t in enumerate(inf_time[:-1]):
                            time_steps.append(t)
                            back_dict[int(t)] = latent
                            time_steps.append(t)
                            if args.save_origin :
                                back_image = pipeline.latents_to_image(latent)[0]
                                back_img_dir = os.path.join(trg_img_output_dir, f'{name}_org_{t}{ext}')
                                back_image.save(back_img_dir)
                            if t == 0:
                                org_input_image = pipeline.latents_to_image(latent)[0].resize((512, 512))
                            noise_pred = call_unet(unet, latent, t, con, None, None)
                            latent = next_step(noise_pred, int(t), latent, scheduler)
                        back_dict[inf_time[-1]] = latent
                        time_steps.append(inf_time[-1])
                        time_steps.reverse() # 999, 750, ..., 0
                        if args.save_origin:
                            back_image = pipeline.latents_to_image(latent)[0]
                            back_img_dir = os.path.join(trg_img_output_dir, f'{name}_org_{inf_time[-1]}{ext}')
                            back_image.save(back_img_dir)

                        #print(f'latent_mask : {latent_mask}')
                        import math
                        def cosine_function(x):
                            x = math.pi * (x - 1)
                            result = math.cos(x)
                            result = result * 0.5
                            result = result + 0.5
                            return result

                        lambda x: cosine_function(x) if x > 0 else 0
                        for i in range(args.inner_iteration) :
                            latent_mask = latent_mask.detach().cpu().apply_(lambda x: cosine_function(x) if x > 0 else 0)
                            latent_mask = latent_mask.to(device)


                        # ----------------------------[3] generate image ------------------------------ #
                        pipeline.to(device)
                        with torch.no_grad():
                            if accelerator.is_main_process:
                                width = height = 512
                                guidance_scale = args.guidance_scale
                                seed = args.seed
                                torch.manual_seed(seed)
                                torch.cuda.manual_seed(seed)
                                height = max(64, height - height % 8)  # round to divisible by 8
                                width = max(64, width - width % 8)  # round to divisible by 8
                                do_classifier_free_guidance = guidance_scale > 1.0
                                x_latent_dict = {}

                                with accelerator.autocast():
                                    text_embeddings = init_prompt(tokenizer, text_encoder, device,args.prompt,args.negative_prompt)
                                    if args.start_from_final :
                                        latents, init_latents_orig, noise = pipeline.prepare_latents(None,None,1,height,width,
                                                                                                     weight_dtype,device,None,None)
                                        if args.start_from_origin:
                                            latents =back_dict[time_steps[0]]
                                        else :
                                            latents = latents
                                        recon_timesteps = time_steps
                                    else :
                                        total_len = len(time_steps) # 980, ,,, , 0
                                        inf_len = int(total_len / 2)
                                        recon_timesteps = time_steps[inf_len:]
                                        latents = back_dict[recon_timesteps[0]]
                                    x_latent_dict[recon_timesteps[0]] = latents
                                    # (7) denoising
                                    for i, t in enumerate(recon_timesteps[1:]) : # 999, 750, ..., 250, 0
                                        # 750, 500, 250, 0
                                        latent_model_input = torch.cat( [latents] * 2) if do_classifier_free_guidance else latents
                                        # predict the noise residual
                                        noise_pred = unet(latent_model_input, t,encoder_hidden_states=text_embeddings).sample
                                        # perform guidance
                                        if do_classifier_free_guidance:
                                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                        latents = pipeline.scheduler.step(noise_pred, t, latents,).prev_sample
                                        if args.use_pixel_mask :
                                            z_latent = back_dict[t]
                                            latents = (z_latent * latent_mask) + (latents * (1 - latent_mask))

                                        if args.only_zero_save :
                                            if t == 0:
                                                image = pipeline.latents_to_image(latents)[0]
                                                img_dir = os.path.join(trg_img_output_dir,f'{name}_recon_{t}{ext}')
                                                image.save(img_dir)
                                        else :
                                            image = pipeline.latents_to_image(latents)[0]
                                            img_dir = os.path.join(trg_img_output_dir, f'{name}_test_final_recon_{t}{ext}')
                                            image.save(img_dir)
                                        controller.reset()
                                    reconstruction = image = pipeline.latents_to_image(latents)[0]
                        # ----------------------------[4] generate anomaly maps ------------------------------ #
                        input = org_input_image
                        input_np = np.array(input)
                        reconstruction_np = np.array(reconstruction)
                        np_diff = np.abs(input_np - reconstruction_np)/255
                        anomaly_maps = np.where(np_diff > 0.4, 1, 0)
                        classification_result = np.sum(anomaly_maps)
                        if classification_result > 0:
                            label = 1
                        else :
                            label = 0
                        Image.fromarray((anomaly_maps*255).astype(np.uint8)).convert('L').save(os.path.join(trg_img_output_dir, f'{name}_anomaly_map{ext}'))




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
    parser.add_argument("--inner_iteration", type=int, default=10)
    parser.add_argument("--use_avg_mask", action='store_true')
    parser.add_argument("--trg_part", type = str)
    parser.add_argument("--only_zero_save", action='store_true')
    parser.add_argument("--truncate_pad", action='store_true')
    parser.add_argument("--truncate_length", type=int, default=3)
    parser.add_argument("--start_from_origin", action='store_true')
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--use_pixel_mask", action='store_true')
    parser.add_argument("--start_from_final", action='store_true')
    parser.add_argument("--save_origin", action='store_true')


    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--cross_map_res", type=arg_as_list, default=[64, 32, 16, 8])
    parser.add_argument("--trg_position", type=arg_as_list, default=['up'])
    parser.add_argument("--trg_lora_epoch", type=str)
    parser.add_argument("--negative_prompt", type=str)

    args = parser.parse_args()
    main(args)