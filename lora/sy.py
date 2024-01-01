import torch

a = torch.randn(8, 1024)
b = torch.randn(8, 1024)
c = torch.cat([a,b], dim=-1)
print(c.shape)
a_, b_ = torch.chunk(c, 2, dim=-1)
print(a_.shape)

normal_score_map = torch.randn(8, 64, 64, 1)
normal_map_total_score = normal_score_map.mean(0).squeeze() #
print(normal_map_total_score.shape)