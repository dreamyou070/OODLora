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
mask[0, 0:pix_num//2] = 1
mask = torch.reshape(mask, (head, int(pix_num ** 0.5), int(pix_num ** 0.5)))[0]# [8, 32,32
print(mask.shape)
image = np.array(mask.cpu().numpy().astype(np.uint8))
np_map = np.array(Image.fromarray(image.astype(np.uint8)).resize((64, 64))) / 255
np_map = np.where(np_map > 0, 1, 0)
mask = torch.from_numpy(np_map)#.unsqueeze(0).unsqueeze(0).float()
print(mask.shape)
masks.append(mask.unsqueeze(0))
masks.append(mask.unsqueeze(0))
out = torch.cat(masks, dim=0)
print(out.shape)
out = out.sum(0) / out.shape[0]
out = 255 * out / out.max()
print(out.shape)