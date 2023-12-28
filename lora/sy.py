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


pixel_set = []
pixel_set.append({64:100})
pixel_set.append({16:32})

ress = [[*elem][0] for elem in pixel_set]
print(ress)
