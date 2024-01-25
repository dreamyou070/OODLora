import torch

a_tensor = torch.tensor([[1.1, 2.1, 3.1],
                         [3.4, 4.1, 1.1]])
size = torch.norm(a_tensor, dim=1, keepdim=True)
a_tensor = a_tensor / size
print(size)
print(a_tensor)

test = 0.2819 * 0.2819 + 0.5381 * 0.5381 + 0.7943 * 0.7943
test = test ** 0.5
print(test)
