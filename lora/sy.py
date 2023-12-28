import os
import numpy as np
from PIL import Image
import torch, torchvision
import torch.nn.functional as nnf
import math
masks = []
pix_num = 32*32
head = 8
mask = torch.zeros((8, pix_num))


masks = []

mask = torch.randn((8,1024*4))
head, pix_num = mask.shape
mask = torch.sum(mask, dim=0)
mask = torch.where(mask > 0, 1, 0) # [1024]
mask = torch.reshape(mask, (int(pix_num ** 0.5), int(pix_num ** 0.5)))
print(mask)
image = np.array(mask.cpu().numpy().astype(np.uint8)) * 255
pil_img = Image.fromarray(image.astype(np.uint8)).resize((64, 64))
np_map = np.array(pil_img) / 255
np_map = np.where(np_map > 0, 1, 0)
print(np_map)
mask = torch.from_numpy(np_map)#.unsqueeze(0).unsqueeze(0).float()
print(mask)
masks.append(mask.unsqueeze(0))

"""
#






out = torch.cat(masks, dim=0) # [num, 64,64]
out = out.sum(0) / out.shape[0]
out = (255 * out / out.max()).unsqueeze(0).unsqueeze(0).float()
mask_latent = out/255
mask_latent = torch.where(mask_latent>0, 1, 0)
"""