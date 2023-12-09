from diffusers import StableDiffusionInpaintPipeline
import torch, os
from PIL import Image
from diffusers import UNet2DConditionModel
import argparse
def main():


    print(f'\n step 1. make model')
    device = 'cuda:1'
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                          cache_dir = r'/data7/sooyeon/pretrained_stable_diffusion').to(device)
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    unet_config_dir = '/data7/sooyeon/pretrained_stable_diffusion/models--runwayml--stable-diffusion-inpainting/snapshots/51388a731f57604945fddd703ecb5c50e8e7b49d/unet/config.json'
    unet_config = UNet2DConditionModel.load_config(pretrained_model_name_or_path=unet_config_dir)
    unet = UNet2DConditionModel.from_config(unet_config).to(device)


if __name__ == "__main__":
    main()