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


mask = torch.randn((8, 1024))
mask = torch.sum(mask, dim=0)
torch.where(mask>0, 1, 0)
mask = torch.reshape(mask, (32,32))
print(mask.shape)
image = np.array(mask.cpu().numpy().astype(np.uint8)) *255
np_map = np.array(Image.fromarray(image.astype(np.uint8)).resize((64, 64))) / 255 # 0 ~ 1
