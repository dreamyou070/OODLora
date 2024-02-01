import torch

a = torch.randn(8,1024, 2)
s = a.mean(dim=0)
print(s.shape)