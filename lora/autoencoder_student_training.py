import argparse
from library import model_util
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
from diffusers.models.vae import DiagonalGaussianDistribution
from generative.losses import PatchAdversarialLoss
import math
import os
import random
import time
import json
from multiprocessing import Value
from tqdm import tqdm
import toml
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library.config_util import (ConfigSanitizer, BlueprintGenerator, )
from library.custom_train_functions import prepare_scheduler_for_custom_training
from STTraining import Encoder_Teacher, Encoder_Student, Decoder_Student, Decoder_Teacher
import torch
from torch.nn import L1Loss
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None


class NetworkTrainer:

    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    """
    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet
    """
    def load_tokenizer(self, args):
        tokenizer = train_util.load_tokenizer(args)
        return tokenizer


    def train(self, args):

        print(f'\n step 1. setting')

        print(f' (1) session')
        if args.process_title:
            setproctitle(args.process_title)
        else:
            setproctitle('parksooyeon')
        session_id = random.randint(0, 2 ** 32)

        print(f' (2) seed')
        if args.seed is None:
            args.seed = random.randint(0, 2 ** 32)
        set_seed(args.seed)


        print(f'\n step 2. dataset')
        tokenizer = self.load_tokenizer(args)
        train_util.prepare_dataset_args(args, True)
        use_class_caption = args.class_caption is not None
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
            print("Using DreamBooth method.")
            user_config = {}
            user_config['datasets'] = [{"subsets": None}]
            subsets_dict_list = []
            for subsets_dict in config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir,args.reg_data_dir,
                                                                                          args.class_caption):
                if use_class_caption:
                    subsets_dict['class_caption'] = args.class_caption
                subsets_dict_list.append(subsets_dict)
                user_config['datasets'][0]['subsets'] = subsets_dict_list
            print(f'User config: {user_config}')
            # blueprint_generator = BlueprintGenerator
            print('start of generate function ...')
            blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
            blueprint.dataset_group
            print(f'blueprint.dataset_group : {blueprint.dataset_group}')
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)
        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            print("No data found. (train_data_dir must be the parent of folders with images) ")
            return


        print(f'\n step 3. preparing accelerator')
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process
        if args.log_with == 'wandb' and is_main_process:
            import wandb
            wandb.init(project=args.wandb_init_name, name=args.wandb_run_name)


        print(f'\n step 4. save directory')
        save_base_dir = args.output_dir
        _, folder_name = os.path.split(save_base_dir)
        record_save_dir = os.path.join(args.output_dir, "record")
        os.makedirs(record_save_dir, exist_ok=True)
        with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)


        print(f'\n step 5. mixed precision and model')
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
        from library.model_util import create_vae_diffusers_config, convert_ldm_vae_checkpoint, load_checkpoint_with_text_encoder_conversion
        from diffusers import AutoencoderKL

        name_or_path = args.pretrained_model_name_or_path
        vae_config = create_vae_diffusers_config()
        _, state_dict = load_checkpoint_with_text_encoder_conversion(name_or_path,
                                                                     device='cpu')
        state_dict = torch.load(args.vae_checkpoint,
                                map_location="cpu")
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

        vae = AutoencoderKL(**vae_config)#.to(device)
        info = vae.load_state_dict(converted_vae_checkpoint)

        """
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)
        """


        vae_encoder = vae.encoder
        vae_encoder_quantize = vae.quant_conv
        vae_encoder.requires_grad_(False)
        vae_encoder_quantize.requires_grad_(False)
        vae_encoder.eval()
        vae_encoder_quantize.eval()
        vae_encoder.to(dtype=weight_dtype)
        vae_encoder_quantize.to(dtype=weight_dtype)

        vae_decoder = vae.decoder
        vae_decoder_quantize = vae.post_quant_conv
        vae_decoder.requires_grad_(False)
        vae_decoder_quantize.requires_grad_(False)
        vae_decoder.eval()
        vae_decoder_quantize.eval()
        vae_decoder.to(dtype=weight_dtype)
        vae_decoder_quantize.to(dtype=weight_dtype)


        teacher_encoder = Encoder_Teacher(vae_encoder, vae_encoder_quantize)
        teacher_decoder = Decoder_Teacher(vae_decoder, vae_decoder_quantize)

        config_dict = vae.config
        from diffusers import AutoencoderKL
        student_vae = AutoencoderKL.from_config(config_dict)
        student_vae_encoder = student_vae.encoder
        student_vae_encoder_quantize = student_vae.quant_conv
        student_encoder = Encoder_Student(student_vae_encoder, student_vae_encoder_quantize)
        student_vae_decoder = student_vae.decoder
        student_vae_decoder_quantize = student_vae.post_quant_conv
        student_decoder = Decoder_Student(student_vae_decoder, student_vae_decoder_quantize)

        teacher_encoder.requires_grad_(False)
        teacher_encoder.eval()
        teacher_encoder.to(dtype=weight_dtype)
        teacher_encoder.to(accelerator.device)

        teacher_decoder.requires_grad_(False)
        teacher_decoder.eval()
        teacher_decoder.to(dtype=weight_dtype)
        teacher_decoder.to(accelerator.device)

        #unet.requires_grad_(False)
        #unet.to(dtype=weight_dtype)
        #for t_enc in text_encoders: t_enc.requires_grad_(False)

        print(f' step 6. dataloader')
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)
        train_dataloader = torch.utils.data.DataLoader(train_dataset_group, batch_size=args.train_batch_size, shuffle=True,
                                                       collate_fn=collater, num_workers=n_workers,
                                                       persistent_workers=args.persistent_data_loader_workers, )
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
            accelerator.print( f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

        print(f'\n step 7. optimizer and lr')
        optimizer = torch.optim.AdamW([{'params' : student_encoder.parameters(),'lr' :1e-4 },
                                       {'params': student_decoder.parameters(), 'lr': 1e-4}],)
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)


        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        print(f'\n step 8. resume')
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)


        if args.resume_vae_training :
            vae_pretrained_dir = args.vae_pretrained_dir
            discriminator_pretrained_dir = args.discriminator_pretrained_dir
            vae.load_state_dict(torch.load(vae_pretrained_dir))


        student_encoder, student_decoder, optimizer, train_dataloader, lr_scheduler= accelerator.prepare(student_encoder, student_decoder, optimizer, train_dataloader, lr_scheduler,)

        # if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
        student_encoder.requires_grad_(True)
        student_encoder.train()
        student_encoder.to(dtype=vae_dtype)

        student_decoder.requires_grad_(True)
        student_decoder.train()
        student_decoder.to(dtype=vae_dtype)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        vae_encoder.to(accelerator.device, dtype=vae_dtype)
        vae_encoder_quantize.to(accelerator.device, dtype=vae_dtype)

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        accelerator.print("running training / 学習開始")
        # accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        # accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        # accelerator.print(f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")
        training_started_at = time.time()
        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            # "ss_num_train_images": train_dataset_group.num_train_images,
            # "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,
            # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
           # "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
        }

        metadata = {}

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name
        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name
        metadata = {k: str(v) for k, v in metadata.items()}
        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]
        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process,
                            desc="steps")
        global_step = 0
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
            clip_sample=False
        )
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
        if accelerator.is_main_process:
            init_kwargs = {}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers("network_train" if args.log_tracker_name is None else args.log_tracker_name,
                                      init_kwargs=init_kwargs)

        del train_dataset_group
        # training loop
        autoencoder_warm_up_n_epochs = args.autoencoder_warm_up_n_epochs
        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1
            metadata["ss_epoch"] = str(epoch + 1)
            student_encoder.train()
            for step, batch in enumerate(train_dataloader):
                log_loss = {}
                # generator training
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    org_img = batch['images'].to(dtype=weight_dtype)
                    masked_img = batch['mask_imgs'].to(dtype=weight_dtype)
                    y = teacher_encoder(masked_img)
                y_hat = student_encoder(org_img)
                loss = torch.nn.functional.mse_loss(y, y_hat, reduction='none')
                loss = loss.mean([1, 2, 3])
                loss = loss.mean()
                log_loss['loss/student_encoder'] = loss.item()

                with torch.no_grad():
                    y_sample = DiagonalGaussianDistribution(y).sample()
                    y_dec = teacher_decoder(y_sample)
                y_hat_dec = student_decoder(DiagonalGaussianDistribution(y_hat).sample())
                loss_dec = torch.nn.functional.mse_loss(y_dec, y_hat_dec, reduction='none')
                loss_dec = loss_dec.mean([1, 2, 3])
                loss_dec = loss_dec.mean()
                log_loss['loss/student_decoder'] = loss_dec.item()
                loss = loss + loss_dec

                if args.student_reconst_loss :
                    batch_size = org_img.shape[0]
                    normal_indexs = []
                    for i in range(batch_size):
                        org = org_img[i]
                        mask = masked_img[i]
                        if torch.equal(org, mask):
                            normal_indexs.append(i)
                    if len(normal_indexs) > 0:
                        normal_org = org_img[normal_indexs]
                        latent = DiagonalGaussianDistribution(y_hat[normal_indexs]).sample()
                        normal_recon = vae_decoder(vae_decoder_quantize(latent))
                        recon_loss = torch.nn.functional.mse_loss(normal_org, normal_recon, reduction='none')
                        recon_loss = recon_loss.mean([1, 2, 3])
                        recon_loss = recon_loss.mean()
                        log_loss['loss/recon'] = recon_loss.mean().item()
                        loss = loss + recon_loss

                # ------------------------------------------------------------------------------------
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if is_main_process:
                        wandb.log(log_loss, step=global_step)
            # ------------------------------------------------------------------------------------------
            if args.save_every_n_epochs is not None:
                print('saving model')
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    trg_epoch = str(epoch + 1).zfill(6)
                    save_directory = os.path.join(args.output_dir, f'vae_encoder_student_model')
                    os.makedirs(save_directory, exist_ok=True)
                    state_dict = student_encoder.state_dict()
                    torch.save(state_dict, os.path.join(save_directory, f'encoder_student_epoch_{trg_epoch}.pth'))

                    save_directory = os.path.join(args.output_dir, f'vae_decoder_student_model')
                    os.makedirs(save_directory, exist_ok=True)
                    torch.save(student_decoder.state_dict(),
                               os.path.join(save_directory, f'decoder_student_epoch_{trg_epoch}.pth'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    # parser.add_argument("--wandb_init_name", type=str)
    parser.add_argument("--wandb_log_template_path", type=str)
    parser.add_argument("--wandb_key", type=str)

    # step 2. dataset
    train_util.add_dataset_arguments(parser, True, True, True)
    parser.add_argument("--mask_dir", type=str, default='')
    parser.add_argument("--class_caption", type=str, default='')
    parser.add_argument("--no_metadata", action="store_true",
                        help="do not save metadata in output model / メタデータを出力先モデルに保存しない")

    # step 3. model
    train_util.add_sd_models_arguments(parser)
    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--network_dim", type=int, default=None,
                        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）")
    parser.add_argument("--network_alpha", type=float, default=1,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)", )
    parser.add_argument("--network_dropout", type=float, default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)", )
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value) / ネットワークへの追加の引数")

    # step 4. training
    train_util.add_training_arguments(parser, True)
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None,
                        help="learning rate for Text Encoder / Text Encoderの学習率")
    # step 5. optimizer
    train_util.add_optimizer_arguments(parser)

    config_util.add_config_arguments(parser)

    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）", )
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
    parser.add_argument("--dim_from_weights", action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する", )
    parser.add_argument("--scale_weight_norms", type=float, default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. ", )
    parser.add_argument("--base_weights", type=str, default=None, nargs="*",
                        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル", )
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*",
                        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う", )
    parser.add_argument("--net_key_names", type=str, default='text')
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--contrastive_eps", type=float, default=0.00005)
    parser.add_argument("--resume_vae_training", action = "store_true")
    parser.add_argument("--perceptual_weight", type = float, default = 0.001)
    parser.add_argument("--autoencoder_warm_up_n_epochs", type=int, default=2)
    parser.add_argument("--vae_pretrained_dir", type=str)
    parser.add_argument("--discriminator_pretrained_dir", type=str)
    parser.add_argument("--student_reconst_loss", action="store_true")
    # class_caption
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    trainer = NetworkTrainer()
    trainer.train(args)