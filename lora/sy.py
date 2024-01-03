import torch


a = torch.randn(8, 32, 3)
attn_list = [a,a]
aa = torch.cat(attn_list, dim=0)
print(aa.shape)