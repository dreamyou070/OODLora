import argparse
from library import model_util
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
from diffusers.models.vae import DiagonalGaussianDistribution
import math
import os
import random
import json
from multiprocessing import Value
from tqdm import tqdm
import toml
from accelerate.utils import set_seed
from library.config_util import (ConfigSanitizer, BlueprintGenerator, )
from STTraining import Encoder_Teacher, Encoder_Student
import torch
from utils.model_utils import get_state_dict
from diffusers import AutoencoderKL
from STTraining import Encoder_Teacher, Encoder_Student, Decoder_Student
from library.model_util import create_vae_diffusers_config, convert_ldm_vae_checkpoint, load_checkpoint_with_text_encoder_conversion
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None


class NetworkTrainer:

    def __init__(self):
        pass

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

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

        print(f' (2) seed')
        if args.seed is None:
            args.seed = random.randint(0, 2 ** 32)
        set_seed(args.seed)

        print(f'\n step 2. dataset')
        tokenizer = self.load_tokenizer(args)
        train_util.prepare_dataset_args(args, True)
        use_class_caption = args.class_caption is not None
        print(f' (2.1) training dataset')
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
            print("Using DreamBooth method.")
            user_config = {}
            user_config['datasets'] = [{"subsets": None}]
            subsets_dict_list = []
            for subsets_dict in config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir,
                                                                                          args.reg_data_dir,
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

        print(f' (2.2) validation dataset')
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        user_config = {}
        user_config['datasets'] = [{"subsets": None}]
        subsets_dict_list = []
        for subsets_dict in config_util.generate_dreambooth_subsets_config_by_subdirs(args.valid_data_dir, args.reg_data_dir, args.class_caption):
            if use_class_caption: subsets_dict['class_caption'] = args.class_caption
            subsets_dict_list.append(subsets_dict)
            user_config['datasets'][0]['subsets'] = subsets_dict_list
        valid_blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        valid_dataset_group = config_util.generate_dataset_group_by_blueprint(valid_blueprint.dataset_group)

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

        print(f'\n step 5. mixed precision and vae model')
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        name_or_path = args.pretrained_model_name_or_path
        vae_config = create_vae_diffusers_config()
        _, state_dict = load_checkpoint_with_text_encoder_conversion(name_or_path, device='cpu')
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)
        vae = AutoencoderKL(**vae_config)
        info = vae.load_state_dict(converted_vae_checkpoint)
        teacher = Encoder_Teacher(vae.encoder, vae.quant_conv)
        teacher.to(dtype=weight_dtype, device=accelerator.device)
        vae.to(dtype=weight_dtype, device=accelerator.device)

        print(f' (5.2) student model')
        config_dict = vae.config
        student_vae = AutoencoderKL.from_config(config_dict)
        student_vae_encoder = student_vae.encoder
        student_vae_encoder_quantize = student_vae.quant_conv
        student = Encoder_Student(student_vae_encoder, student_vae_encoder_quantize)

        print(f' step 6. dataloader')
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)
        train_dataloader = torch.utils.data.DataLoader(train_dataset_group, batch_size=args.train_batch_size, shuffle=True,
                                                       collate_fn=collater, num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers, )
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset_group, batch_size=args.train_batch_size, shuffle=False,
                                                       collate_fn=collater, num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers, )
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
            accelerator.print( f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

        print(f'\n step 7. optimizer and lr')
        optimizer = torch.optim.AdamW([{'params' : student.parameters(),'lr' :1e-4 },],)
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        print(f'\n step 8. resume')
        if args.resume_vae_training :
            encoder_state_dict = get_state_dict(args.student_pretrained_dir)
            student.load_state_dict(encoder_state_dict, strict=True)

        student, optimizer, train_dataloader, valid_dataloader, lr_scheduler= accelerator.prepare(student, optimizer, train_dataloader, valid_dataloader, lr_scheduler,)

        student.requires_grad_(True)
        student.train()
        student.to(dtype=weight_dtype)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        print(f'\n step 9. training')

        accelerator.print("running training")
        accelerator.print(f"  num batches per epoch : {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process,desc="steps")
        global_step = 0
        if accelerator.is_main_process:
            init_kwargs = {}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers("network_train" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs)
        del train_dataset_group

        # training loop
        for epoch in range(args.start_epoch, args.start_epoch + num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1
            student.train()
            """

            for step, batch in enumerate(train_dataloader):
                log_loss = {}
                # generator training
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    org_img = batch['images'].to(dtype=weight_dtype)
                    masked_img = batch['mask_imgs'].to(dtype=weight_dtype)
                    y = teacher(masked_img)
                y_hat = student(org_img)
                loss = torch.nn.functional.mse_loss(y, y_hat, reduction='none')
                loss = loss.mean([1, 2, 3])
                loss = loss.mean()
                log_loss['loss/student_encoder'] = loss.item()

                if args.only_normal_training :
                #if args.student_reconst_loss :
                    batch_size = org_img.shape[0]
                    normal_indexs = []
                    for i in range(batch_size):
                        org = org_img[i]
                        mask = masked_img[i]
                        if torch.equal(org, mask):
                            normal_indexs.append(i)

                    if len(normal_indexs) > 0:
                        normal_org = org_img[normal_indexs]
                        with torch.no_grad():
                            y = teacher(normal_org)
                        y_hat = student(normal_org)
                        loss = torch.nn.functional.mse_loss(y, y_hat, reduction='none')
                        loss = loss.mean([1, 2, 3])
                        loss = loss.mean()
                        log_loss['loss/student_encoder_normal'] = loss.item()

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
                    save_directory = os.path.join(args.output_dir, f'vae_student_model')
                    os.makedirs(save_directory, exist_ok=True)
                    state_dict = student.state_dict()
                    torch.save(state_dict,
                               os.path.join(save_directory, f'student_epoch_{trg_epoch}.pth'))
                    # inference
            with torch.no_grad():
                if is_main_process:
                    img = batch['images'].to(dtype=weight_dtype)
                    latent = DiagonalGaussianDistribution(student(img)).sample()
                    recon = vae.decode(latent)['sample']
                    batch = recon.shape[0]
                    if batch != 1:
                        recon = recon[0]
                        recon = recon.unsqueeze(0)
                    recon_img = (recon / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
                    import numpy as np
                    image = (recon_img * 255).astype(np.uint8)
                    wandb.log({"recon": [wandb.Image(image, caption="recon")]})
            # --------------------------------------------------------------------------------------------------------- #
            """
            # validation
            valid_epoch_normal_loss = 0
            valid_epoch_abnormal_loss = 0
            for step, valid_batch in enumerate(valid_dataloader):
                student.eval()
                with torch.no_grad():
                    org_img = valid_batch['images'].to(dtype=weight_dtype)
                    y = DiagonalGaussianDistribution(teacher(org_img)).sample()
                    y_recon = vae.decode(y)['sample']

                    y_hat = DiagonalGaussianDistribution(student(org_img)).sample()
                    y_hat_recon = vae.decode(y_hat)['sample']

                    binary_images = valid_batch['binary_images'].to(dtype=weight_dtype)



                    print(f'y : {y.shape}')
                    print(f'y_recon : {y_recon.shape}')
                    print(f'binary images : {binary_images.shape}')

                    normal_recon_diff = torch.nn.functional.mse_loss(y_recon, y_hat_recon, reduction='none') * binary_images
                    normal_recon_diff = normal_recon_diff.mean([1, 2, 3])
                    abnormal_recon_diff = torch.nn.functional.mse_loss(y_recon, y_hat_recon, reduction='none') * (1 - binary_images)
                    abnormal_recon_diff = abnormal_recon_diff.mean([1, 2, 3])
                    wandb.log({'valid/normal_loss' : normal_recon_diff.mean().item(),
                               'valid/abnormal_loss' : abnormal_recon_diff.mean().item()})
                    valid_epoch_normal_loss += normal_recon_diff.mean().item()
                    valid_epoch_abnormal_loss += abnormal_recon_diff.mean().item()
            valid_log = {'normal_loss' : valid_epoch_normal_loss,
                         'anormal_loss' : valid_epoch_abnormal_loss,}
            if is_main_process:
                accelerator.log(valid_log, step=epoch + 1)

            # --------------------------------------------------------------------------------------------------------- #
            # [3] model save
            if args.save_every_n_epochs is not None:
                print('saving model')
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    trg_epoch = str(epoch + 1).zfill(6)
                    save_directory = os.path.join(args.output_dir, f'vae_student_model')
                    os.makedirs(save_directory, exist_ok=True)
                    state_dict = student.state_dict()
                    torch.save(state_dict,
                               os.path.join(save_directory, f'student_epoch_{trg_epoch}.pth'))
                    # inference
            with torch.no_grad():
                if is_main_process:
                    img = batch['images'].to(dtype=weight_dtype)
                    latent = DiagonalGaussianDistribution(student(img)).sample()
                    recon = vae.decode(latent)['sample']
                    batch = recon.shape[0]
                    if batch != 1:
                        recon = recon[0]
                        recon = recon.unsqueeze(0)
                    recon_img = (recon / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
                    import numpy as np
                    image = (recon_img * 255).astype(np.uint8)
                    wandb.log({"recon": [wandb.Image(image, caption="recon")]})

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
    # step 3. model
    train_util.add_sd_models_arguments(parser)
    # step 4. training
    train_util.add_training_arguments(parser, True)
    custom_train_functions.add_custom_train_arguments(parser)
    # step 5. optimizer
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    parser.add_argument("--valid_data_dir", type=str)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--contrastive_eps", type=float, default=0.00005)
    parser.add_argument("--resume_vae_training", action = "store_true")
    parser.add_argument("--perceptual_weight", type = float, default = 0.001)
    parser.add_argument("--autoencoder_warm_up_n_epochs", type=int, default=2)
    parser.add_argument("--student_pretrained_dir", type=str)
    parser.add_argument("--only_normal_training", action="store_true")
    parser.add_argument("--start_epoch", type = int)
    # class_caption
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    trainer = NetworkTrainer()
    trainer.train(args)