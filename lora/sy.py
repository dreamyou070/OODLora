import os
import numpy as np
from PIL import Image
import torch, torchvision
import torch.nn.functional as nnf
import math

mask = torch.randn((8, 32*32))
cross_maps = mask.reshape(8, 32,32)
map_list = [cross_maps,cross_maps]
out = torch.cat(map_list, dim=0)  # [batch*head, res,res]
avg_attn = out.sum(0) / out.shape[0]
print(avg_attn.shape)
out = out.mean(dim=0) # [res,res]
print(out.shape)
print(torch.equal(avg_attn,out))
image = out
image = 255 * image / image.max() # res,res,4
image = image.unsqueeze(-1).expand(*image.shape, 4).cpu()  # res,res,3
print(image.shape)
image = image.numpy().astype(np.uint8)
image = np.array(Image.fromarray(image).resize((64,64))) # high 255