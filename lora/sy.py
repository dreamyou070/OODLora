import torch

a = torch.randn(8, 1024)
map_list = [a,a,a,a,a]
map = torch.cat(map_list, dim=0)
map = map.float().mean([0])
map = map.reshape(32,32)
print(map.shape)