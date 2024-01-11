import os
from torchvision import transforms
from PIL import Image
import torch

anormal_mask_dir = 'crack_000.png'

anormal_mask_64 = transforms.ToTensor()(Image.open(anormal_mask_dir).convert('L').resize((64, 64), Image.BICUBIC))
anormal_mask_32 = transforms.ToTensor()(Image.open(anormal_mask_dir).convert('L').resize((32, 32), Image.BICUBIC))
anormal_mask_16 = transforms.ToTensor()(Image.open(anormal_mask_dir).convert('L').resize((16, 16), Image.BICUBIC))
anormal_mask_8 = transforms.ToTensor()(Image.open(anormal_mask_dir).convert('L').resize((8, 8), Image.BICUBIC))

anormal_mask_8_ = torch.where(anormal_mask_8==0, 0, 1).float()

print(anormal_mask_8)
print(anormal_mask_8_)