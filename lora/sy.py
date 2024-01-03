import torch



mask = torch.randn((32,32))
mask = torch.stack([mask.flatten() for i in range(8)], dim=0).unsqueeze(-1) # 8, 32*32, 1
print(mask.shape)