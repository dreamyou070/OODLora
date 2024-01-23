import torch

a = torch.tensor([0.9, 1.2, 0.3])
a_clip = torch.clamp(a, min=0, max=1.0)
print(a_clip)