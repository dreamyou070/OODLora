from torch.nn import L1Loss, MSELoss
import torch
import collections


org_img = torch.randn((3,3,512,512))
mask_img = torch.randn((3,3,512,512))
batch_size = org_img.shape[0]
normal_indexs = []
for i in range(batch_size):
    org = org_img[i]
    mask = mask_img[i]
    if torch.equal(org,mask) :
        normal_indexs.append(i)



normal_recon = org_img[[]]
print(normal_recon.shape)