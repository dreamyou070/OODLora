import torch

torch_a = torch.tensor([[1., 2.], [3, 4.]])
torch_b = torch.tensor([[1., 2.], [3,4.]])

answer = torch.equal(torch_a, torch_b)
print(answer)