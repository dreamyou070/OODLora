import torch
from torch import nn
from torchvision import transforms
import math
import numpy as np


avg_attn_map_ls = []
for i in range(3):
    a = torch.randn(8, 32,32,1).mean(0)
    print(a.shape)
    avg_attn_map_ls.append(a)
avg_attn_map = torch.stack(avg_attn_map_ls, dim=0)
print(avg_attn_map.shape)
# 3, 32, 32, 1