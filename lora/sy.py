import numpy as np
import torch
timestep_np = np.array([957, 924, 891, 858, 825, 792, 759, 726, 693, 660, 627, 594, 561, 528,
        495, 462, 429, 396, 363, 330, 297, 264, 231, 198, 165, 132,  99,  66,
         33,   0])
timestep = torch.from_numpy(timestep_np)

for i in torch.flip(timestep, dims=[0]):
    print(i)
