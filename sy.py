import torch
import numpy as np


#
"""
total_image_pred = np.array([])
batch, h,w = 3, 128,128
pred_mask = torch.randn((batch, 1, h,w))
out_mask = pred_mask
topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
image_score = torch.mean(topk_out_mask)
total_image_pred = np.append(total_image_pred, image_score.detach().cpu().numpy())
image_score = torch.mean(topk_out_mask)
total_image_pred = np.append(total_image_pred, image_score.detach().cpu().numpy())
print(total_image_pred)
"""
from PIL import Image

orgin_latent = torch.randn((3,4,64,64))
orgin_latent = torch.flatten(orgin_latent, start_dim=1)
orgin_latent_np = orgin_latent.detach().cpu().numpy()
print(orgin_latent_np.shape)