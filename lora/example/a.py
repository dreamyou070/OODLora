from PIL import Image
import numpy as np
import torch

a = torch.randn((1,32,32,1))
attn_score = torch.randn((8,32,32,1))

a = a.expand(attn_score.shape)
print(a.shape)