import torch




ca_map_obj_1 = torch.randn((8,8,8,77))
print(ca_map_obj_1.shape)
a = ca_map_obj_1[:,:,:,1:5].softmax(dim=-1)
print(a.shape)