import torch

a= torch.randn((8,32))
b = a[:,:2]
print(b.shape)