import os
import numpy as np
from PIL import Image
import torch


mask_latent = torch.randn((64,64))
z_noise_pred = torch.randn((1,4,64,64))
mask_latent = mask_latent.unsqueeze(0).unsqueeze(0)
print(f'mask_latent shape : {mask_latent.shape}')
mask_latent = mask_latent.expand(z_noise_pred.shape)
