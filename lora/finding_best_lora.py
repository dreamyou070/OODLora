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
from utils.image_utils import image2latent, load_image
from utils.scheduling_utils import get_scheduler
from utils.model_utils import init_prompt
import shutil
from attention_store import AttentionStore
import torch.nn as nn

try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None


def register_attention_control(unet: nn.Module, controller: AttentionStore,
                               mask_thredhold: float = 1):  # if mask_threshold is 1, use itself

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
            if is_cross_attention and trg_indexs_list is not None:

                cls_score  = attention_probs[:, :, 0]
                good_score = attention_probs[:, :, 1]
                bad_score  = attention_probs[:, :, 2]
                scores = torch.cat([cls_score, good_score, bad_score], dim=-1)
                res = int((good_score.shape[1]) ** 0.5)
                if res in args.cross_map_res:
                    #score_diff = good_score - bad_score
                    controller.store(scores, layer_name)


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

    parent = os.path.split(args.network_weights)[0]
    args.output_dir = os.path.join(parent, 'anormality_score')

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

    trg_resolutions = args.cross_map_res
    title = ''
    for res in trg_resolutions:
        title += f'_{res}'
    print(f'title : {title}')

    os.makedirs(output_dir, exist_ok=True)
    print(f'final output dir : {output_dir}')

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
    if args.dim_from_weights:
        network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet,
                                                                **net_kwargs)
    else:
        network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet,
                                                neuron_dropout=args.network_dropout, **net_kwargs, )
    print(f' (2.5.3) apply trained state dict')
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        parent, epoch = os.path.split(args.network_weights)
        epoch, ext = os.path.splitext(epoch)
        epoch = int(epoch.split("-")[-1])

    network.to(device)
    controller = AttentionStore()
    register_attention_control(unet, controller, mask_thredhold=args.mask_thredhold)

    print(f' \n step 3. ground-truth image preparing')
    print(f' (3.1) prompt condition')
    prompt = 'good bad'
    context = init_prompt(tokenizer, text_encoder, device, prompt)
    uncon, con = torch.chunk(context, 2)
    uncon, con = uncon[:, :3, :], con[:, :3, :]
    print(f'uncon : {uncon.shape} | con : {con.shape}')
    context = torch.cat([uncon, con], dim=0)

    print(f' (3.2) train images')
    trg_h, trg_w = args.resolution

    print(f' (3.3) test images')
    test_img_folder = os.path.join(args.concept_image_folder, 'test_ex/bad')
    test_mask_folder = os.path.join(args.concept_image_folder, 'test_ex/corrected')
    classes = os.listdir(test_img_folder)
    test_output_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    record_file = os.path.join(test_output_dir, f'lora_epoch_{epoch}_score_diff_record.txt')
    lines = []
    for class_name in classes:
        class_base_folder = os.path.join(test_output_dir, class_name)
        #os.makedirs(class_base_folder, exist_ok=True)

        image_folder = os.path.join(test_img_folder, class_name)
        mask_folder = os.path.join(test_mask_folder, class_name)

        invers_context = init_prompt(tokenizer, invers_text_encoder, device, f'a photo of {class_name}')
        inv_unc, inv_c = invers_context.chunk(2)
        test_images = os.listdir(image_folder)

        for j, test_image in enumerate(test_images):
            name, ext = os.path.splitext(test_image)
            #trg_img_output_dir = os.path.join(class_base_folder, f'{name}')
            #os.makedirs(trg_img_output_dir, exist_ok=True)

            test_img_dir = os.path.join(image_folder, test_image)
            #shutil.copy(test_img_dir, os.path.join(trg_img_output_dir, test_image))

            # if 'good' not in class_name:
            mask_img_dir = os.path.join(mask_folder, test_image)
            #shutil.copy(mask_img_dir, os.path.join(trg_img_output_dir, f'{name}_mask{ext}'))


            print(f' (2.3.1) inversion')
            image_gt_np = load_image(test_img_dir, trg_h=int(trg_h), trg_w=int(trg_w))

            with torch.no_grad():
                latent = image2latent(image_gt_np, vae, device=device, weight_dtype=weight_dtype)

            from utils.model_utils import call_unet
            with torch.no_grad():
                # con = [CLS, Good, Bad]
                noise_pred = call_unet(unet, latent, 0, con, [[1]], None)
                map_dict = controller.step_store
                controller.reset()

                cls_score_list, good_score_list, bad_score_list = [], [], []
                for layer in map_dict.keys():
                    scores = map_dict[layer][0]
                    cls_score, good_score, bad_score = scores.chunk(3, dim=-1)
                    cls_score_list.append(cls_score)
                    good_score_list.append(good_score)
                    bad_score_list.append(bad_score)
                cls_score = torch.cat(cls_score_list, dim=0).mean(dim=0).squeeze().reshape(int(args.cross_map_res[0]),int(args.cross_map_res[0])) # [res*res]
                good_score = torch.cat(good_score_list, dim=0).mean(dim=0).squeeze().reshape(int(args.cross_map_res[0]),int(args.cross_map_res[0])) # [res*res]
                bad_score = torch.cat(bad_score_list, dim=0).mean(dim=0).squeeze().reshape(int(args.cross_map_res[0]),int(args.cross_map_res[0])) # [res*res]
                total_score = cls_score + good_score + bad_score
                print(f'cls_score : {cls_score}')
                print(f'good_score : {good_score}')
                print(f'bad_score : {bad_score}')
                print(f'total_score : {total_score}')

                #score_diff = torch.cat(score_list, dim=0).mean(dim=0).squeeze() # [res*res]
                #print(f'score_diff : {score_diff}')

                mask_pil = Image.open(mask_img_dir).convert('L')
                mask_pil = mask_pil.resize((int(args.cross_map_res[0]),int(args.cross_map_res[0])), Image.BICUBIC)
                mask_np = np.array(mask_pil, np.uint8)
                mask_np = np.where(mask_np > 10, 1, 0)  # binary mask
                mask_np = torch.tensor(mask_np, dtype=torch.float32, device=device)

                anormal_position = mask_np # 16,16
                normal_position =  1- mask_np

                print(f'cls score of anormal_position : {cls_score*anormal_position}')
                print(f'good score of anormal_position : {good_score*anormal_position}')
                import time
                time.sleep(50)


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