import numpy as np
import random
import torch
from PIL import Image

"""

"""
mask_torch = torch.randn((8,8*8,1))
head, pix_num, i = mask_torch.shape
res = int(pix_num**0.5)
cross_maps = mask_torch.reshape(head, res, res,i)
cross_maps = cross_maps.mean([-1])
cross_maps = cross_maps.mean([0])
import torchvision
image = cross_maps.numpy().astype(np.uint8)
totensor = torchvision.transforms.ToTensor()
image = totensor(Image.fromarray(image).resize((64, 64)))