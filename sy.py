import torch


import math
import numpy as np

pixel_mask = torch.tensor([[0,1],
                           [0,0]])

anormal_mask = torch.tensor([[1,0],
                           [0,0]])

normal_position = torch.where((pixel_mask == 1 ) & (anormal_mask == 0), 1, 0)
print(normal_position)
