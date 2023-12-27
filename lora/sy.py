import os
import numpy as np
from PIL import Image
import torch


mask1 = torch.randn((1, 64, 64))
mask2 = torch.randn((1, 64, 64))
mask3 = torch.randn((1, 64, 64))
mask4 = torch.randn((1, 64, 64))
mask5 = torch.randn((1, 64, 64))
masks = [mask1, mask2, mask3, mask4, mask5]
mask_latent = torch.cat(masks, dim=0)
mask_latent = mask_latent.mean(dim=0, dtype=torch.float32)
print(mask_latent.shape)
mask_latent = mask_latent.unsqueeze(0).unsqueeze(0)
z_noise_pred = torch.randn((1, 4, 64, 64))
mask_latent = mask_latent.expand(z_noise_pred.shape)
print(mask_latent.shape)