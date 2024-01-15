import torch


import math
import numpy as np

img_mask = torch.tensor([[1,1],
                           [0,0]])

anormal_mask = torch.tensor([[1,0],
                           [0,0]])

back_position = torch.where(img_mask == 0, 1, 0)
anormal_position = torch.where((anormal_mask == 1), 1, 0)
normal_position = torch.where((back_position == 0) & (anormal_position == 0), 1, 0)
total_position = back_position + anormal_position + normal_position
print(f'back_position: {back_position}')
print(f'anormal_position: {anormal_position}')
