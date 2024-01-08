import torch

key = torch.randn(8, 30)
key = torch.cat([key, key], dim=0)
print(key.shape)