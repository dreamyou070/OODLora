import torch

a = torch.randn(8, 1024)
b = torch.randn(8, 1024)
c = torch.cat([a,b], dim=-1)
#print(c.shape)
a_, b_ = torch.chunk(c, 2, dim=-1)
#print(a_.shape)

normal_mask_ = torch.randn((1,64,64))
normal_mask_ = normal_mask_.repeat(8, 1, 1) # [h, res, res]
print(normal_mask_.shape)