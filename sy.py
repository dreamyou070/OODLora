import torch

torch_a = torch.randn((4,64,64))
a = [torch_a, torch_a]

img = a[0]
print(img.shape)
b= torch.stack(a, dim=0)
print(b.shape)