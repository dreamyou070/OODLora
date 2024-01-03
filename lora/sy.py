import torch



score_map = torch.randn((8, 32*32))
score_per_heard = score_map.sum(dim=-1)
print(score_per_heard.shape)