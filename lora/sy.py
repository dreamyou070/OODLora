import torch


a = torch.tensor([[1,2,3,4],
              [4,5,6,7]])
a_sum = torch.sum(a, dim=0)/a.shape[0]
# 5, 7, 9, 11
print(a_sum)