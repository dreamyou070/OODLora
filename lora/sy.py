import torch

vector = torch.randn((8, 4096))
vectors = [vector,vector,vector,vector,vector]
maps = torch.cat(vectors, dim=0)
maps = maps.sum(0) / maps.shape[0]
maps = maps.reshape(64, 64)
print(maps.shape)