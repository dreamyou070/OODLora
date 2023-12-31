import torch

binary_aug_tensor = torch.randn((1,64,64,1))
a = [binary_aug_tensor,binary_aug_tensor]
a = torch.cat(a, dim=0)
print(a.shape)