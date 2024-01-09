import torch
import numpy as np

latent_mask_np = np.array([1,2,3])
latent_mask_torch = torch.from_numpy(latent_mask_np)
print(latent_mask_torch)