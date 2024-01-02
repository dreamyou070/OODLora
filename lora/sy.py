import torch


a = torch.tensor([[1,2,3,4],
              [4,5,6,7]])
a_sum = torch.sum(a, dim=0)/a.shape[0]
# 5, 7, 9, 11
print(a_sum)

ca_map_obj_1 = torch.randn((8,8,8))
ca_map_obj_2 = torch.randn((8,8,8))
maps = []
maps.append(ca_map_obj_1)
maps.append(ca_map_obj_2)
maps = torch.cat(maps, dim=0)
print(maps.shape)