import torch
object_position = torch.randn((8, 1024))
object_position = object_position.sum(dim=0)
head = 0
object_position = torch.where(object_position > head, 1, 0).unsqueeze(0)
object_position = object_position.expand((8, 1024))
print(object_position.shape)