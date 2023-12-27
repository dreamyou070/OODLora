import os
import numpy as np
from PIL import Image
import torch

masked_attn_vector = torch.randn((8,64,1))
org_attn_vector = torch.randn((8,64,1))
diff = torch.nn.functional.mse_loss(masked_attn_vector, org_attn_vector, reduction = 'none')
print(diff.shape)
sum = torch.sum(diff, 0)
print(sum.shape)
#   diff = torch.sum(diff, axis=1)
#print(diff)