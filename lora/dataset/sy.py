from torchvision import transforms
from PIL import Image
import torch
import torchvision
from torch import nn


normal_score_map_i = torch.randn((8, 32,32))
normal_total_score = normal_score_map_i.reshape(8, -1).sum(dim=-1)
print(normal_total_score.shape)