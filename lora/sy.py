import torch

a_torch = torch.tensor([[0.4,0.54],
                        [0.03,0.03]])
flatten_a = torch.flatten(a_torch)
m = torch.nn.Softmax()
b = m(flatten_a)
b = b.reshape(2,2)
print(b.sum())
print(b)