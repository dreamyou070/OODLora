from torchvision import transforms
from PIL import Image
import torch
import torchvision
from torch import nn

normal_score = torch.tensor([[0.8, 0.1],
                             [0.9, 0.9]])
# big to big
answer = torch.tensor([[1., 0.], [1., 1.]])
answer2 = torch.tensor([[0., 1.], [0., 0.]])
bce_loss_func = nn.BCELoss()
a = bce_loss_func(normal_score, answer)
b = bce_loss_func(normal_score, answer2)
print(a)
print(b)
normal_score_map_i = torch.randn(8,8,8)
b, H, W, = normal_score_map_i.shape
a = normal_score_map_i.mean(0)
print(a.shape)