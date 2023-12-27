import os
import numpy as np
from PIL import Image
import torch


map = torch.ones((8, 32,32))
map_list = [map, map]
map = torch.cat(map_list, dim=0).mean(dim=0)
np_map = np.array(map) * 255
aug_map = Image.fromarray(np_map).resize((64,64))
np_aug_map = np.array(aug_map)
torch_aug_map = torch.tensor(np_aug_map)/255

print(torch_aug_map)

