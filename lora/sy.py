import torch

probs = torch.randn((8, 32*32,77))
crack = probs[:, :, 0].unsqueeze(-1)
hole = probs[:, :, 1].unsqueeze(-1)
print(crack.shape)
maps = torch.cat([crack, hole], dim=-1)
print(maps.shape)