from torch.nn import L1Loss, MSELoss
import torch
import collections


org_img = torch.randn((3,3,512,512))
a = torch.Tensor([1])
print(a.shape)
