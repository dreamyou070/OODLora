import argparse
from accelerate.utils import set_seed
# from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
import os
import random
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
import torch
from PIL import Image
import sys, importlib
from utils.image_utils import image2latent, load_image
from utils.scheduling_utils import get_scheduler
from utils.model_utils import get_state_dict, init_prompt
from attention_store import AttentionStore
import torch.nn as nn
from utils.model_utils import call_unet
from utils.scheduling_utils import next_step
import math
from utils.common_utils import get_lora_epoch, save_latent
from scipy.spatial.distance import mahalanobis
from utils.model_utils import get_crossattn_map
from utils.image_utils import latent2image, numpy_to_pil
from safetensors.torch import load_file

try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None



def get_position(layer_name, attn):
    res = int(attn.shape[1] ** 0.5)
    if 'down' in layer_name:
        pos = 'down'
    elif 'up' in layer_name:
        pos = 'up'
    else:
        pos = 'mid'
    if 'attentions_0' in layer_name:
        part = 'attn_0'
    elif 'attentions_1' in layer_name:
        part = 'attn_1'
    else:
        part = 'attn_2'
    return res, pos, part


def register_attention_control(unet: nn.Module, controller: AttentionStore,
                               mask_threshold: float = 1):  # if mask_threshold is 1, use itself

    def ca_forward(self, layer_name):

        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            self_head_query = self.to_q(hidden_states) # batch, pix_num, dim
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)
            query = self.reshape_heads_to_batch_dim(self_head_query)
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
                controller.store(attention_probs[:, :, :args.truncate_length], layer_name)
                if layer_name == 'up_blocks_3_attentions_0_transformer_blocks_0_attn2':
                    controller.save_query(self_head_query, layer_name)


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


def main(args):

    parent = os.path.split(args.network_weights)[0]  # unique_folder,
    args.output_dir = os.path.join(parent, 'reconstruction_20240127')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f' \n step 1. setting')
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f'\n step 2. preparing accelerator')
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
    tokenizer = train_util.load_tokenizer(args)
    device = accelerator.device

    print(f'\n step 3. save directory and save config')
    parent = os.path.split(args.network_weights)[0]
    center_folders = os.path.join(parent, 'centers')

    network_weights = os.listdir(args.network_weights)

    for weight in network_weights:
        name = os.path.splitext(weight)[0]
        lora_epoch = int(name.split('-')[-1])

        back_file = os.path.join(center_folders, f'background_{lora_epoch}.pt')
        normal_file = os.path.join(center_folders, f'normal_{lora_epoch}.pt')
        import pickle
        with open(back_file, 'rb') as f:
            background= pickle.load(f)
        with open(normal_file, 'rb') as f:
            normal = pickle.load(f)
        b_center, b_cov = background
        n_center, n_cov = normal
        print(f'background center: {b_center.shape}, normal center: {n_center.shape}')


        if accelerator.is_main_process:

            # (1) call basic model
            text_encoder, vae, unet, _ = train_util._load_target_model(args, weight_dtype, device,
                                                                       unet_use_linear_projection_in_v2=False, )
            unet, text_encoder = unet.to(device), text_encoder.to(device)
            vae.to(accelerator.device, dtype=vae_dtype)
            scheduler_cls = get_scheduler(args.sample_sampler, args.v_parameterization)[0]
            scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps,
                                      beta_start=args.scheduler_linear_start,
                                      beta_end=args.scheduler_linear_end,
                                      beta_schedule=args.scheduler_schedule)

            # (2) make scratch network
            sys.path.append(os.path.dirname(__file__))
            network_module = importlib.import_module(args.network_module)
            net_kwargs = {}
            network = network_module.create_network(1.0, args.network_dim, args.network_alpha, None,
                                                    text_encoder, unet,
                                                    neuron_dropout=args.network_dropout,
                                                    **net_kwargs, )
            network.apply_to(text_encoder, unet, True, True)
            network.restore()
            # (3) save direction
            model_epoch = get_lora_epoch(weight)
            test_lora_dir = os.path.join(args.output_dir, f'lora_{model_epoch}')
            os.makedirs(test_lora_dir, exist_ok=True)
            ## (3.1) base dir
            condition_save_dir = os.path.join(test_lora_dir, f'memory_bank')
            print(f'condition_save_dir : {condition_save_dir}')
            os.makedirs(condition_save_dir, exist_ok=True)
            ## (3.2) evaluate dir
            evaluate_output_dir = os.path.join(condition_save_dir,
                                               f'{args.class_name}/test')
            os.makedirs(evaluate_output_dir, exist_ok=True)
            ## (3.3) my inference dir
            my_inference_output_dir = os.path.join(condition_save_dir,
                                                   f'my_inference_lora_{model_epoch}')
            os.makedirs(my_inference_output_dir, exist_ok=True)

            # (4) get images
            test_folder = os.path.join(args.concept_image_folder, 'test')
            classes = os.listdir(test_folder)

            normal_vectors, background_vectors = [], []
            for class_name in classes:
                flag = True
                if args.only_normal_infer:
                    if 'good' not in class_name:
                        flag = False
                if flag:
                    evaluate_class_dir = os.path.join(evaluate_output_dir, class_name)
                    os.makedirs(evaluate_class_dir, exist_ok=True)
                    class_base_folder = os.path.join(my_inference_output_dir, class_name)
                    os.makedirs(class_base_folder, exist_ok=True)
                    image_folder = os.path.join(test_folder, f'{class_name}/rgb')
                    mask_folder = os.path.join(test_folder, f'{class_name}/gt')
                    test_images = os.listdir(image_folder)

                    for j, test_image in enumerate(test_images):

                        print(f' {class_name} : {test_image}')
                        name, ext = os.path.splitext(test_image)
                        test_img_dir = os.path.join(image_folder, test_image)
                        mask_img_dir = os.path.join(mask_folder, test_image)
                        org_h, org_w = Image.open(mask_img_dir).size
                        Image.open(mask_img_dir).convert('L').resize((org_h, org_w)).save(
                            os.path.join(class_base_folder, f'{name}_gt{ext}'))
                        if accelerator.is_main_process:
                            with torch.no_grad():
                                org_img = load_image(test_img_dir, 512, 512)
                                org_vae_latent = image2latent(org_img, vae, device, weight_dtype)
                                # ------------------------------------- [1] object mask ------------------------------ #
                                weight_dir = os.path.join(args.network_weights, weight)
                                network.load_weights(weight_dir)
                                network.to(device)
                                controller_ob = AttentionStore()
                                register_attention_control(unet, controller_ob)
                                with torch.no_grad():
                                    context_ob = init_prompt(tokenizer, text_encoder, device, args.prompt)
                                uncon_ob, con_ob = torch.chunk(context_ob, 2)
                                call_unet(unet, org_vae_latent, 0, con_ob[:, :args.truncate_length, :], None, None)
                                query_dict = controller_ob.query_dict
                                controller_ob.reset()
                                # ------------------------------------- [2] make latent mask ------------------------------ #
                                # network.restore()
                                features = query_dict['up_blocks_3_attentions_0_transformer_blocks_0_attn2'][0].squeeze() # pix_num, dim
                                pix_num = features.size(0)
                                res = int(pix_num ** 0.5)
                                """
                                pdist = torch.nn.PairwiseDistance(p=2)
                                n_vector = normal_vector.unsqueeze(0).repeat(pix_num, 1)
                                b_vector = back_vector.unsqueeze(0).repeat(pix_num, 1)
                                n_diff = pdist(features, n_vector)
                                b_diff = pdist(features, b_vector)
                                """
                                n_dist_list, b_dist_list, t_dist_list = [], [], []
                                for s in range(pix_num) :
                                    sample = features[s,:].squeeze().cpu()
                                    n_dist = mahalanobis(sample, n_center, n_cov)
                                    b_dist = mahalanobis(sample, b_center, b_cov)
                                    total_dist = n_dist + b_dist
                                    n_dist = n_dist / total_dist
                                    n_dist_list.append(n_dist)
                                n_dist_vector = torch.stack(n_dist_list, dim=0).unsqueeze(0)
                                n_dist_map = n_dist_vector.reshape(res,res)
                                recon_mask = n_dist_map.to(device)
                                #total_diff = n_diff + b_diff
                                #n_diff = n_diff / total_diff
                                #b_diff = b_diff / total_diff
                                #diff = torch.where(n_diff > b_diff, n_diff, b_diff) #
                                #anomal_mask = diff.unsqueeze(0)
                                #recon_mask = anomal_mask.reshape(res,res)
                                print(f'recon_mask : {recon_mask}')
                                recon_mask_save_dir = os.path.join(class_base_folder, f'{name}_recon_mask{ext}')
                                save_latent(recon_mask, recon_mask_save_dir, org_h, org_w)
                                recon_mask = (recon_mask.unsqueeze(0).unsqueeze(0)).repeat(1, 4, 1, 1)

                                # ---------------------------------- [2] reconstruction ------------------------------ #
                                from utils.pipeline import AnomalyDetectionStableDiffusionPipeline
                                import numpy as np
                                pipeline = AnomalyDetectionStableDiffusionPipeline(vae=vae,
                                                                                   text_encoder=text_encoder,
                                                                                   tokenizer=tokenizer,
                                                                                   unet=unet,
                                                                                   scheduler=scheduler,
                                                                                   safety_checker=None,
                                                                                   feature_extractor=None,
                                                                                   requires_safety_checker=False, )
                                latents = pipeline(prompt=args.prompt,
                                                   height=512,
                                                   width=512,
                                                   num_inference_steps=args.num_ddim_steps,
                                                   guidance_scale=args.guidance_scale,
                                                   negative_prompt=args.negative_prompt,
                                                   reference_image=org_vae_latent,
                                                   mask=recon_mask)
                                controller_ob.reset()
                                # mask = None)
                                recon_image = pipeline.latents_to_image(latents)[0].resize((org_h, org_w))
                                img_dir = os.path.join(class_base_folder, f'{name}_recon{ext}')
                                recon_image.save(img_dir)

                                # -------------------------------------- [4] anomaly map -------------------------------------- #
                                #network.restore()
                                #network.load_weights(weight_dir)
                                #network.to(device)
                                #controller = AttentionStore()
                                #register_attention_control(unet, controller)
                                with torch.no_grad():
                                    context = init_prompt(tokenizer, text_encoder, device, args.prompt)
                                uncon, con = torch.chunk(context, 2)
                                # (1) original
                                org_img = pipeline.latents_to_image(org_vae_latent)[0].resize((org_h, org_w))
                                org_img.save(os.path.join(class_base_folder, f'{name}_org{ext}'))
                                call_unet(unet, org_vae_latent, 0, con, None, 1)
                                org_query = \
                                controller_ob.query_dict['up_blocks_3_attentions_0_transformer_blocks_0_attn2'][0].squeeze(0)
                                org_query = org_query / (torch.norm(org_query, dim=1, keepdim=True))
                                controller_ob.reset()

                                # (2) recon
                                recon_latent = latents
                                call_unet(unet, recon_latent, 0, con, None, 1)
                                recon_query = \
                                controller_ob.query_dict['up_blocks_3_attentions_0_transformer_blocks_0_attn2'][0].squeeze(0)
                                controller_ob.reset()
                                recon_query = recon_query / (torch.norm(recon_query, dim=1, keepdim=True))

                                # (3) anomaly score
                                print(f'org_query : {org_query.shape}')
                                anomaly_score = (org_query @ recon_query.T).cpu()
                                print(f'anomaly_score : {anomaly_score.shape}')
                                pix_num = anomaly_score.shape[0]
                                anomaly_score = (torch.eye(pix_num) * anomaly_score).sum(dim=0)
                                anomaly_score = anomaly_score / anomaly_score.max()  # 0 ~ 1
                                anomaly_score = anomaly_score.unsqueeze(0).reshape(64, 64)
                                anomaly_score = anomaly_score.numpy()

                                anomaly_score_pil = Image.fromarray((255 - (anomaly_score * 255)).astype(np.uint8))
                                anomaly_score_pil = anomaly_score_pil.resize((org_h, org_w))
                                anomaly_mask_save_dir = os.path.join(class_base_folder, f'{name}{ext}')
                                tiff_anomaly_mask_save_dir = os.path.join(evaluate_class_dir, f'{name}.tiff')
                                anomaly_score_pil.save(anomaly_mask_save_dir)
                                anomaly_score_pil.save(tiff_anomaly_mask_save_dir)




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
    parser.add_argument("--network_dropout", type=float, default=None, )
    parser.add_argument("--network_args", type=str, default=None, nargs="*", )
    parser.add_argument("--dim_from_weights", action="store_true", )
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--concept_image", type=str,
                        default='/data7/sooyeon/MyData/perfusion_dataset/td_100/100_td/td_1.jpg')
    parser.add_argument("--prompt", type=str, default='teddy bear, wearing like a super hero')
    parser.add_argument("--concept_image_folder", type=str)
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--scheduler_linear_start", type=float, default=0.00085)
    parser.add_argument("--scheduler_linear_end", type=float, default=0.012)
    parser.add_argument("--scheduler_timesteps", type=int, default=1000)
    parser.add_argument("--scheduler_schedule", type=str, default="scaled_linear")
    parser.add_argument("--inner_iteration", type=int, default=10)
    parser.add_argument("--class_name", type=str, default="bagel")
    parser.add_argument("--free_time", type=int, default=80)
    parser.add_argument("--only_zero_save", action='store_true')
    parser.add_argument("--truncate_pad", action='store_true')
    parser.add_argument("--truncate_length", type=int, default=3)
    parser.add_argument("--start_from_origin", action='store_true')
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--use_pixel_mask", action='store_true')
    parser.add_argument("--start_from_final", action='store_true')
    parser.add_argument("--save_origin", action='store_true')
    parser.add_argument("--only_normal_infer", action='store_true')
    parser.add_argument("--latent_diff_thred", type=float, default=0.5)
    parser.add_argument("--anormal_thred", type=float, default=0.5)
    parser.add_argument("--detection_network_weights", type=str, )
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--cross_map_res", type=arg_as_list, default=[64, 32, 16, 8])
    parser.add_argument("--trg_position", type=arg_as_list, default=['up'])
    parser.add_argument("--trg_part", type=arg_as_list, default=['attn_2', 'attn_1', 'attn_0'])
    parser.add_argument("--trg_lora_epoch", type=str)
    parser.add_argument("--negative_prompt", type=str)

    args = parser.parse_args()
    main(args)