import torch

a = torch.randn(8,64*64)
map_list = [a,a,a,a,a]
aa = torch.cat(map_list, dim=0).mean(dim=0)
pix_num = aa.shape
res = 64
map = aa.reshape(res,res)
print(map.shape)