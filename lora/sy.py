import os
import torch
a = torch.randn((3,4))
a = a.unsqueeze(1)
print(a.shape)