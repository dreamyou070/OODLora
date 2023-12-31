import torch

a= torch.randn((2, 16, 16, 4))
aa = a.sum(dim=[1,2,3])
print(aa)