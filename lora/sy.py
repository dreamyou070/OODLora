from torch.nn import L1Loss, MSELoss
import torch
import collections


org_img = torch.randn((3,3,512,512))
a = torch.Tensor([1])
print(a.shape)
torch.manual_seed(0)
d = torch.randn_like(org_img)


torch.manual_seed(0)
e = torch.randn_like(org_img)

print(torch.equal(d,e))
