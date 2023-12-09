# VAE model
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/autoencoder_kl_32x32x4.yaml -t --gpus 0,
# LDM model
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/inpaint.yaml -t --gpus 0,