from torchvision import transforms
from PIL import Image
import torch
import torchvision
from torch import nn


mask = torch.randn((4,4))
mask = torch.stack([mask for i in range(8)], dim=0) #
print(mask.shape)