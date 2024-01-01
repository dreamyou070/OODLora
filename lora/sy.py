import torch

from torchvision import transforms
res = 2
resize_transform = transforms.Resize((res, res))
a = torch.ones(1, 1, 4,4)
b = resize_transform(a)
print(a)
print(b)