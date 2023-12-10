import torch

a = [torch.randn((3)),
     torch.randn((3))]
a = torch.stack(a, dim=0)
a = a.mean(dim=0)
loss = a.mean()
print(a)