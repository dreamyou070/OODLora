from torchvision import transforms
from PIL import Image
import torch
import torchvision
from torch import nn


mask = torch.randn((4,4))
a = mask[:,1:3]
print(a.shape)