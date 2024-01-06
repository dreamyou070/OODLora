import torch


import math


def cosine_function(x):
    x = math.pi * (x-1)
    result = math.cos(x)
    result = result * 0.5
    result = result + 0.5
    return result

mask_torch = torch.tensor([[0.3,0.4]])

lambda x: cosine_function(x) if x > 0 else 0
mask_torch = mask_torch.apply_(lambda x: cosine_function(x) if x > 0 else 0)
print(mask_torch)