import torch, ast

res_ = 32
a = torch.randn(8, res_*res_)
a_list = [a,a,a]
attn  = torch.cat(a_list, dim=0).unsqueeze(-1) # 24,32,1
#attn = attn.mean(dim=0)
#print(attn.shape)
h = attn.shape[0]
attn = attn.reshape(h, res_, res_).float().mean(dim=0).unsqueeze(0).unsqueeze(0) # almost binary
attn = attn.repeat(1, 4, 1, 1)
