import torch

object_position = torch.tensor([0.9, 1.2, 0.3])


object_position = torch.triu(object_position, min = 0, max = 1)