import torch

a = torch.randn(8, 1024)
b = torch.randn(8, 1024)
c = torch.cat([a,b], dim=-1)
#print(c.shape)
a_, b_ = torch.chunk(c, 2, dim=-1)
#print(a_.shape)
import torch.nn as nn
input = torch.tensor([[0.9,0.1],
                      [0.7,0.3],
                      [0.8,0.2]]).float()
#input = nn.Sigmoid()(input)
print(input)

answer_1 = torch.tensor([[1.0,0.0],
                         [1.0,0.0],
                         [1.0,0.0]]).float()
#cross_entropy = torch.nn.BCELoss()

#loss_1 = cross_entropy(input, answer_1)
loss_1 = torch.nn.BCELoss()(input, answer_1)
print(loss_1)