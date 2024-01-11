import os
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
anormal_mask_dir = f'train_ex_2/gt/combined_000.png'
anormal_mask_64 = transforms.ToTensor()(Image.open(anormal_mask_dir).convert('L').resize((64, 64), Image.BICUBIC))
anormal_mask_64 = torch.where(anormal_mask_64 == 0, 0, 1).float()
mask = anormal_mask_64

pixel_mask_dir= f'train_ex_2/mask/combined_000.png'
pixel_mask_64 = transforms.ToTensor()(Image.open(pixel_mask_dir).convert('L').resize((64, 64), Image.BICUBIC))
pixel_mask_64 = torch.where(pixel_mask_64 == 1, 1, 0).float()
img_mask = pixel_mask_64
normal_mask = torch.where((img_mask - mask) == 1, 1, 0).squeeze(0)
pil = Image.fromarray(np.array(normal_mask).astype(np.uint8) * 255)
pil.show()
