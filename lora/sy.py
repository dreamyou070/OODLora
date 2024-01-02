import torch




ca_map_obj_1 = torch.randn((8,8))

res = [ca_map_obj_1,ca_map_obj_1]
score_map = torch.cat(res, dim=0)
score_map = score_map.float().mean(dim=0).squeeze().reshape(8,8)
print(score_map.shape)