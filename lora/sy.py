import torch

a = torch.randn(8, 1024)
b = torch.randn(8, 1024)
c = torch.cat([a,b], dim=-1)
print(c.shape)
a_, b_ = torch.chunk(c, 2, dim=-1)
print(a_.shape)