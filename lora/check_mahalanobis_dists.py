import importlib, argparse, gc, math, os, sys, random, time, json, toml
from PIL import Image
from accelerate.utils import set_seed
from library import model_util
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
import torch
from torch import nn
from attention_store import AttentionStore
from utils.model_utils import call_unet
from torchvision import transforms
from utils.image_utils import load_image
import matplotlib.pyplot as plt
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None

os.environ["WANDB__SERVICE_WAIT"] = "500"


def register_attention_control(unet: nn.Module, controller: AttentionStore,
                               mask_threshold: float = 1):  # if mask_threshold is 1, use itself

    def ca_forward(self, layer_name):
        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True

            query = self.to_q(hidden_states)  # batch, pix_num, dim
            controller.save_query(query, layer_name)

            # ---------------------------------------------------------------------------------------------------------
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
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            if is_cross_attention and trg_indexs_list is not None:
                if args.cls_training:
                    trg_map = attention_probs[:, :, :2]
                else:
                    trg_map = attention_probs[:, :, 1]
                controller.store(trg_map, layer_name)

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

class NetworkTrainer:

    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    def generate_step_logs(self, loss_dict, lr_scheduler, keys_scaled=None, mean_norm=None, maximum_norm=None,
                           **kwargs):
        logs = {}
        for k, v in loss_dict.items():
            logs[k] = v
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
                    args.optimizer_type.lower().startswith(
                        "DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                        lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0][
                    "lr"])
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith(
                        "DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                            lr_scheduler.optimizers[-1].param_groups[i]["d"] *
                            lr_scheduler.optimizers[-1].param_groups[i]["lr"])

        return logs

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def extract_triggerword_index(self, input_ids):
        cls_token = 49406
        pad_token = 49407
        if input_ids.dim() == 3:
            input_ids = torch.flatten(input_ids, start_dim=1)
        batch_num, sen_len = input_ids.size()
        batch_index_list = []

        for batch_index in range(batch_num):
            token_ids = input_ids[batch_index, :].squeeze()
            index_list = []
            for index, token_id in enumerate(token_ids):
                if token_id != cls_token and token_id != pad_token:
                    index_list.append(index)
            batch_index_list.append(index_list)
        return batch_index_list

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)  # batch, torch_num, sen_len
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids,
                                                             tokenizers[0], text_encoders[0],
                                                             weight_dtype)
        return encoder_hidden_states

    def get_class_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["class_input_ids"].to(accelerator.device)  # batch, torch_num, sen_len
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids,
                                                             tokenizers[0], text_encoders[0],
                                                             weight_dtype)
        return encoder_hidden_states

    def call_unet(self,
                  args, accelerator, unet,
                  noisy_latents, timesteps,
                  text_conds, batch, weight_dtype,
                  trg_indexs_list,
                  mask_imgs):
        noise_pred = unet(noisy_latents,
                          timesteps,
                          text_conds,
                          trg_indexs_list=trg_indexs_list,
                          mask_imgs=mask_imgs, ).sample
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def train(self, args):

        args.logging_dir = os.path.join(args.output_dir, 'logs')

        print(f'\n step 1. setting')
        print(f' (1) session')
        if args.process_title:
            setproctitle(args.process_title)
        else:
            setproctitle('parksooyeon')
        print(f' (2) seed')
        if args.seed is None: args.seed = random.randint(0, 2 ** 32)
        set_seed(args.seed)

        print(f'\n step 2. dataset')
        tokenizer = train_util.load_tokenizer(args)

        print(f'\n step 3. preparing accelerator')
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        print(f'\n step 4. save directory')
        save_base_dir = args.output_dir
        _, folder_name = os.path.split(save_base_dir)
        parent, network_dir = os.path.split(args.network_weights)
        name, ex = os.path.splitext(network_dir)
        lora_epoch = int(name.split('-')[-1])
        record_save_dir = os.path.join(args.output_dir, f"record_lora_eopch_{lora_epoch}")
        os.makedirs(record_save_dir, exist_ok=True)

        print(f' (4.1) config saving')
        with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        print(f'\n step 5. model')
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
        print(f' (5.1) base model')
        model_version, frozen_text_encoder, vae, frozen_unet = self.load_target_model(args, weight_dtype,
                                                                                          accelerator)
        train_util.replace_unet_modules(frozen_unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        print(' (5.2) lora model')
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)
        net_kwargs = {}
        frozen_network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae,
                                 frozen_text_encoder, frozen_unet, neuron_dropout=args.network_dropout, **net_kwargs, )

        print(' (5.3) lora with unet and text encoder')
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = not args.network_train_unet_only
        frozen_network.apply_to(frozen_text_encoder, frozen_unet, train_text_encoder, train_unet)
        frozen_network.load_weights(args.network_weights)

        # -------------------------------------------------------------------------------------------------------- #
        print(f'\n step 6. make memory bank')
        device = accelerator.device
        frozen_unet, frozen_text_encoder, network = frozen_unet.to(device),frozen_text_encoder.to(device),\
            frozen_network.to(device)

        class Mahalanobis_dataset(torch.utils.data.Dataset):
            def __init__(self, train_data_dir):

                self.vae_scale_factor = 0.18215
                self.image_transforms = transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize([0.5], [0.5]), ])

                self.imgs = []
                parent, rgb = os.path.split(train_data_dir)  # train_normal, rgb
                mask_base_dir = os.path.join(parent, 'mask')
                class_names = os.listdir(train_data_dir)
                for class_name in class_names:
                    rgb_class_dir = os.path.join(train_data_dir, class_name)
                    mask_class_dir = os.path.join(mask_base_dir, class_name)
                    images = os.listdir(rgb_class_dir)
                    for img in images:
                        rgb_dir = os.path.join(rgb_class_dir, img)
                        mask_dir = os.path.join(mask_class_dir, img)
                        self.imgs.append((rgb_dir, mask_dir, class_name))

            def __len__(self):
                return len(self.imgs)

            def __getitem__(self, idx):

                rgb_dir, mask_dir, class_name = self.imgs[idx]
                # latent
                img = load_image(rgb_dir, 512, 512)  # ndarray, dim=3
                torch_img = self.image_transforms(img).unsqueeze(0)  # dim = 4, torch image
                with torch.no_grad():
                    latent = vae.encode(torch_img.to(vae.device, dtype=vae_dtype)).latent_dist.sample()
                    latent = latent * self.vae_scale_factor
                # mask
                img_masks = transforms.ToTensor()(Image.open(mask_dir).
                                                  convert('L').resize((64, 64), Image.BICUBIC))  # [64,64]
                object_mask = torch.where(img_masks > 0.5, 1, 0).float().squeeze()  # res,res
                sample = {}
                sample['latent'] = latent
                sample['object_mask'] = object_mask
                sample['class_name'] = class_name
                return sample

        mahal_dataset = Mahalanobis_dataset(args.all_data_dir)
        controller = AttentionStore()
        register_attention_control(frozen_unet, controller, mask_threshold=args.mask_threshold)
        with torch.no_grad():
            text_input = tokenizer([args.class_caption], padding="max_length",
                                   max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt", )
            text = text_input.input_ids.to(device)
            text_embeddings = frozen_text_encoder(text)[0][:, :2, :]

        normal_vector_list = set()
        normal_vector_good_score_list, normal_vector_bad_score_list = set(), set()
        back_vector_list = set()
        anormal_vector_list = set()
        num_samples = len(mahal_dataset)
        for i in range(num_samples):
            sample = mahal_dataset.__getitem__(i)
            class_name = sample['class_name']
            checking = False
            if args.do_check_anormal :
                checking = True
            else :
                if 'good' in class_name:
                    checking = True
            if checking :
                latent = sample['latent']  # 1,4,64,64
                if latent.dim() == 3:
                    latent = latent.unsqueeze(0)
                latent = latent.to(device)  # 1,4,64,64
                mask = sample['object_mask']  # res,res
                mask_vector = mask.flatten()  # pix_num
                with torch.no_grad():
                    if text_embeddings.dim() != 3:
                        text_embeddings = text_embeddings.unsqueeze(0)
                    call_unet(frozen_unet, latent, 0, text_embeddings.to(device)[:,:2,:], 1, args.trg_layer_list)
                    layer_1 = args.trg_layer_list[0]
                    if args.concat_query :
                        layer_2 = args.trg_layer_list[1]
                        query_1 = controller.query_dict[layer_1][0].squeeze()  # pix_num, dim
                        query_2 = controller.query_dict[layer_2][0].squeeze()  # pix_num, dim
                        query = torch.cat([query_1, query_2], dim=-1)  # pix_num, 2*dim
                    else :
                        query = controller.query_dict[layer_1][0].squeeze()  # pix_num, dim

                    if args.cls_training :
                        attn = controller.step_store[layer_1][0].squeeze()  # 1, pix_num, 2
                        cls_map, trigger_map = attn.chunk(2, dim=-1)
                    else :
                        attn = controller.step_store[layer_1][0].squeeze()  # 1, pix_num, 1
                        trigger_map = attn.squeeze()
                    trigger_map = trigger_map.squeeze()
                    trigger_map = trigger_map.mean(dim=0) # pix_num

                if 'good' in class_name:
                    for pix_idx in range(mask_vector.shape[0]):
                        feature = query[pix_idx, :].cpu()
                        attn_score = trigger_map[pix_idx].cpu()
                        if mask_vector[pix_idx] == 1:
                            if feature.dim() == 1:
                                feature = feature.unsqueeze(0)
                            if type(attn_score) == torch.Tensor:
                                attn_score = attn_score.item()
                            if attn_score > 0.5 :
                                normal_vector_good_score_list.add(feature)
                                normal_vector_list.add(feature)
                            else :
                                normal_vector_bad_score_list.add(feature)
                                normal_vector_list.add(feature)
                        else:
                            if feature.dim() == 1:
                                feature = feature.unsqueeze(0)
                            back_vector_list.add(feature)
                else:
                    for pix_idx in range(mask_vector.shape[0]):
                        feature = query[pix_idx, :].cpu()
                        if mask_vector[pix_idx] == 1:
                            if feature.dim() == 1:
                                feature = feature.unsqueeze(0)
                            anormal_vector_list.add(feature)
                if i % 20 == 0:
                    print(f'normal_vector_good_score_list : {len(normal_vector_good_score_list)}, '
                          f'normal_vector_bad_score_list : {len(normal_vector_bad_score_list)}, '
                          f'anormal : {len(anormal_vector_list)}')
        # ----------------------------------------------------------------------------------------------------------- #
        def mahal(u, v, cov):
            delta = u - v
            m = torch.dot(delta, torch.matmul(cov, delta))
            return torch.sqrt(m)

        if len(normal_vector_good_score_list) > 0:
            normal_vector_good_score = torch.cat(list(normal_vector_good_score_list), dim=0)  # sample, dim
            normal_vector_mean_torch = torch.mean(normal_vector_good_score, dim=0)
            normal_vectors_cov_torch = torch.cov(normal_vector_good_score.transpose(0, 1))
        else :
            normal_vectors = torch.cat(list(normal_vector_list), dim=0)  # sample, dim
            normal_vector_mean_torch = torch.mean(normal_vectors, dim=0)
            normal_vectors_cov_torch = torch.cov(normal_vectors.transpose(0, 1))


        # ----------------------------------------------------------------------------------------------------------- #
        # [1] good mahalanobis distances
        if len(normal_vector_good_score_list) > 0:
            mahalanobis_dists = [mahal(feat, normal_vector_mean_torch, normal_vectors_cov_torch) for
                                 feat in normal_vector_good_score]
            plt.figure()
            plt.hist(mahalanobis_dists)
            save_dir = os.path.join(record_save_dir, "normal_goodscore_mahalanobis_distances.png")
            plt.savefig(save_dir)
            save_dir = os.path.join(record_save_dir, "normal_goodscore_mahalanobis_distances.txt")
            with open(save_dir, 'w') as f:
                for d in mahalanobis_dists :
                    f.write(f'{d},')
        else :
            mahalanobis_dists = [mahal(feat, normal_vector_mean_torch, normal_vectors_cov_torch) for
                                 feat in normal_vectors]
            plt.figure()
            plt.hist(mahalanobis_dists)
            save_dir = os.path.join(record_save_dir, "normal_mahalanobis_distances.png")
            plt.savefig(save_dir)
            save_dir = os.path.join(record_save_dir, "normal_mahalanobis_distances.txt")
            with open(save_dir, 'w') as f:
                for d in mahalanobis_dists:
                    f.write(f'{d},')
        # ----------------------------------------------------------------------------------------------------------- #
        if args.do_check_anormal:
            anormal_vectors = torch.cat(list(anormal_vector_list), dim=0)  # sample, dim
            a_mahalanobis_dists = [mahal(feat, normal_vector_mean_torch, normal_vectors_cov_torch) for
                                 feat in anormal_vectors]
            plt.figure()
            plt.hist(a_mahalanobis_dists)
            save_dir = os.path.join(record_save_dir, "anormal_mahalanobis_distances.png")
            plt.savefig(save_dir)
            save_dir = os.path.join(record_save_dir, "anormal_mahalanobis_distances.txt")
            with open(save_dir, 'w') as f:
                for d in a_mahalanobis_dists:
                    f.write(f'{d},')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. setting
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    parser.add_argument("--wandb_log_template_path", type=str)
    parser.add_argument("--wandb_key", type=str)
    # step 2. dataset
    train_util.add_dataset_arguments(parser, True, True, True)
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--class_caption", type=str, default='')
    # step 3. model
    train_util.add_sd_models_arguments(parser)
    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--network_dim", type=int, default=None,
                        help="network dimensions (depends on each network)")
    parser.add_argument("--network_alpha", type=float, default=1,
               help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)", )
    parser.add_argument("--network_dropout", type=float, default=None,
           help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)", )
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument("--scale_weight_norms", type=float, default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. ", )
    parser.add_argument("--base_weights", type=str, default=None, nargs="*",
                        help="network weights to merge into the model before training", )
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*",
                        help="multiplier for network weights to merge into the model before training", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision",)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--truncate_pad", action='store_true')
    parser.add_argument("--truncate_length", type=int, default=3)
    parser.add_argument("--all_data_dir", type=str)
    parser.add_argument("--concat_query", action='store_true')
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--trg_part", type=arg_as_list, default=['down', 'up'])
    parser.add_argument("--trg_layer", type=str)
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=['down_blocks_0_attentions_1_transformer_blocks_0_attn2'])
    parser.add_argument("--average_mask", action="store_true", )
    parser.add_argument("--normal_with_background", action="store_true", )
    parser.add_argument("--only_object_position", action="store_true", )
    parser.add_argument("--do_check_anormal", action="store_true", )
    parser.add_argument("--seed",type=int, default=42)
    args = parser.parse_args()
    trainer = NetworkTrainer()
    trainer.train(args)