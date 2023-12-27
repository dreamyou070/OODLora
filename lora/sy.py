import os
import numpy as np
from PIL import Image
import torch

map_list = [torch.zeros((8,32,32))]
map = torch.cat(map_list, dim=0)
print(f'before mean, map : {map.shape}')
map = map.mean([0])
print(f'before mean, map : {map.shape}')
np_map = np.array(map) * 255
aug_map = Image.fromarray(np_map).resize((64, 64))
np_aug_map = np.array(aug_map)
torch_aug_map = torch.tensor(np_aug_map)
mask = torch.where(torch_aug_map > 100, 1, 0)
#controller.store(mask, layer_name)
#attention_probs = torch.cat([attention_probs_back, attention_probs_object_sub], dim=0)
#hidden_states = torch.bmm(attention_probs, value)