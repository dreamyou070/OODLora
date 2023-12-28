import argparse
from diffusers.models.vae import DiagonalGaussianDistribution
import os
import random
from accelerate.utils import set_seed
import library.train_util as train_util
import library.config_util as config_util
import torch
from utils.image_utils import load_image, IMAGE_TRANSFORMS
import numpy as np
from scipy.spatial.distance import mahalanobis
from utils.model_utils import get_state_dict
import pickle
from STTraining import Encoder_Teacher, Encoder_Student, Decoder_Student
from library.model_util import create_vae_diffusers_config, convert_ldm_vae_checkpoint, load_checkpoint_with_text_encoder_conversion
try:
    from setproctitle import setproctitle
except (ImportError, ModuleNotFoundError):
    setproctitle = lambda x: None
from accelerate import Accelerator
from diffusers import AutoencoderKL


def main(args):

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

    print(f' (3) accelerator')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,)

    print(f'\n step 2. Teacher VAE')
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32
    name_or_path = args.pretrained_model_name_or_path
    vae_config = create_vae_diffusers_config()
    _, state_dict = load_checkpoint_with_text_encoder_conversion(name_or_path, device='cpu')
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)
    vae = AutoencoderKL(**vae_config)
    info = vae.load_state_dict(converted_vae_checkpoint)
    vae.to(dtype=weight_dtype, device=accelerator.device)
    vae.eval()

    print(f'\n step 3. Teacher Model Mean & Covariance')
    os.makedirs(args.output_dir, exist_ok=True)
    object = os.path.split(args.anormal_folder)[-1]
    train_feature_filepath = os.path.join(args.output_dir, f'vae_teacher_{object}.pkl')

    h, w = args.resolution.split(',')[0], args.resolution.split(',')[1]
    if not os.path.exists(train_feature_filepath):
        train_folder = os.path.join(args.anormal_folder, 'train_normal/bad')
        classes = os.listdir(train_folder)
        training_latents = []
        for class_ in classes:
            class_dir = os.path.join(train_folder, class_)
            images = os.listdir(class_dir)
            for i, image in enumerate(images):
                image_dir = os.path.join(class_dir, image)
                img = load_image(image_dir, int(h.strip()), int(w.strip()))
                img = IMAGE_TRANSFORMS(img).to(dtype=vae_dtype).unsqueeze(0)
                with torch.no_grad():
                    latent = vae.encode(img.to(dtype=weight_dtype, device=accelerator.device)).latent_dist.sample()
                training_latents.append(latent)
        embedding_vectors = torch.cat(training_latents, dim=0)
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).detach().cpu()  # [N, 550, 3136]
        # (1) mean vector
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        print(f'mean vector (4, 64*64) : {mean.shape}')
        # (2) covariance vector
        cov = torch.zeros(C, C, H * W).numpy()
        print(f'covariance vector (4,4,64*64) : {cov.shape}')
        I = np.identity(C)
        for i in range(H * W):
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        train_outputs = [mean, cov]
        os.makedirs(args.output_dir, exist_ok=True)
        object = os.path.split(args.anormal_folder)[-1]
        train_feature_filepath = os.path.join(args.output_dir, f'vae_teacher_{object}.pkl')
        print(f'feature file path : {train_feature_filepath}')
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)
    else:
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)

    print(f'\n step 4. Student VAE')
    student_vae = AutoencoderKL.from_config(vae.config)
    student_vae_encoder = student_vae.encoder
    student_vae_encoder_quantize = student_vae.quant_conv
    student_encoder = Encoder_Student(student_vae_encoder, student_vae_encoder_quantize)

    parent, folder = os.path.split(args.student_encoder_pretrained_dir)
    model_score_save_base_dir = os.path.join(parent, f'vae_student_model_score')
    os.makedirs(model_score_save_base_dir, exist_ok=True)

    student_pretrained_dir_files = os.listdir(args.student_encoder_pretrained_dir)
    for file in student_pretrained_dir_files:
        name, ext = os.path.splitext(file)
        txt_file = os.path.join(model_score_save_base_dir, f'{name}.txt')
        file_dir = os.path.join(args.student_encoder_pretrained_dir, file)
        student_encoder.load_state_dict(get_state_dict(file_dir), strict=True)
        student_encoder.requires_grad_(False)
        student_encoder.eval()
        student_encoder.to(accelerator.device, dtype=vae_dtype)
        name, ext = os.path.splitext(file)
        student_epoch = int(name.split('_')[-1])
        # ------------------------------------------------------------------------------------ #
        test_folder = os.path.join(args.anormal_folder, 'test/rgb')
        cats = os.listdir(test_folder)
        for cat in cats :
            cat_dir = os.path.join(test_folder, cat)
            images = os.listdir(cat_dir)
            if len(images) > 0 :
                latents = []
                for i, image in enumerate(images):
                    image_dir = os.path.join(cat_dir, image)
                    img = load_image(image_dir, int(h.strip()), int(w.strip()))
                    img = IMAGE_TRANSFORMS(img).to(dtype=vae_dtype).unsqueeze(0)
                    with torch.no_grad():
                        latent = DiagonalGaussianDistribution(student_encoder(img.to(dtype=weight_dtype, device=accelerator.device))).sample() # 1,4,64,64
                        latents.append(latent)
                embedding_vectors = torch.cat(latents, dim=0)
                # calculate distance matrix
                B, C, H, W = embedding_vectors.size()
                embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()  # [N, 550, 3136]

                img_level_dist = []
                for i in range(H * W):
                    mean = train_outputs[0][:, i]
                    conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
                    pixel_level_dist = []
                    for sample in embedding_vectors: # for B number
                        m_dist = mahalanobis(sample[:, i], mean, conv_inv)
                        pixel_level_dist.append(m_dist)
                    img_level_dist.append(pixel_level_dist)
                score_map = np.array(img_level_dist).transpose(1, 0).reshape(B, H, W)
                model_score = np.sum(np.mean(score_map, axis=0))
                with open(txt_file, 'a') as f:
                    f.write(f'{cat} : {model_score}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    train_util.add_sd_models_arguments(parser)
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    config_util.add_config_arguments(parser)
    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）", )
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--resolution", type=str, default=None, help="resolution in training ('size' or 'width,height') ", )
    parser.add_argument("--logging_dir",type=str,default=None,)
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"],)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int,default=1,)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--full_fp16", action="store_true",
                        help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients / 勾配も含めてbf16で学習する")
    parser.add_argument("--save_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--lowram",action="store_true",
                        help="enable low RAM optimization. e.g. load models to VRAM instead of RAM ",)
    parser.add_argument("--vae", type=str, default=None,
                        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")
    parser.add_argument("--anormal_sample_data_dir", type=str,
                        default=r'../../../MyData/anomaly_detection/VisA/MVTecAD/bagel/test/crack/rgb/000.png')
    parser.add_argument("--anormal_folder", type=str,)
    parser.add_argument("--normal_sample_data_dir", type=str,
                        default=r'../../../MyData/anomaly_detection/VisA/MVTecAD/bagel/test/good/rgb/000.png')
    parser.add_argument("--student_encoder_pretrained_dir", type=str,)
    parser.add_argument("--student_decoder_pretrained_dir", type=str, )
    parser.add_argument("--training_data_check", action="store_true",)
    args = parser.parse_args()
    main(args)