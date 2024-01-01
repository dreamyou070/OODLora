from torchvision import transforms
from PIL import Image
import torch
import torchvision
from torch import nn

normal_score = torch.tensor([[0.8, 0.1],
                             [0.9, 0.9]])
# big to big
answer = torch.tensor([[1., 0.],
                       [1., 1.]])
out_auto = torch.nn.BCELoss(reduction = 'none')(normal_score, answer)
print(out_auto)
answer = normal_score
out_auto = torch.nn.BCELoss(reduction = 'none')(normal_score, answer)
print(out_auto)
out = answer * torch.log(normal_score) + (1 - answer) * torch.log(1 - normal_score)
out_man = -1 * out
#print(f'manual output : {out_man}')




anormal = 1-answer
out_auto_anormal = torch.nn.BCELoss(reduction = 'none')(normal_score, anormal)
print(out_auto_anormal)

