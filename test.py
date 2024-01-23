import torch

object_position = torch.tensor([0.9, 1.2, 0.3])


object_position = torch.clamp(object_position, min = 0, max = 1)
print(object_position)