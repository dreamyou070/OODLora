import os
from PIL import Image
import numpy as np
import torch

latent = torch.randn((1, 4, 64, 64))

input_latents = torch.cat([latent, latent], dim=0)
print(input_latents.shape)
"""

rgb_dir = r'/data7/sooyeon/MyData/anomaly_detection/MVTecAD/bad'
mask_dir = r'/data7/sooyeon/MyData/anomaly_detection/MVTecAD/bad_mask'
masked_dir = r'/data7/sooyeon/MyData/anomaly_detection/MVTecAD/bad_masked'
os.makedirs(masked_dir, exist_ok=True)
classes = os.listdir(rgb_dir)
for cls in classes:
    rgb_cls_dir = os.path.join(rgb_dir, cls)
    mask_cls_dir = os.path.join(mask_dir, cls)
    masked_dir_cls = os.path.join(masked_dir, cls)
    os.makedirs(masked_dir_cls, exist_ok=True)
    images = os.listdir(rgb_cls_dir)
    mask_images = os.listdir(mask_cls_dir)
    for i in images:
        img = Image.open(os.path.join(rgb_cls_dir, i))
        mask = Image.open(os.path.join(mask_cls_dir, i))
        mask_np = np.array(mask)
        img_np = np.array(img)
        img_np[mask_np > 200] = 0
        Image.fromarray(img_np).save(os.path.join(masked_dir_cls, i))
"""
"""        
mask_dir= r"gt"
rgb_dir = r"rgb"
masked_dir = r"masked"
os.makedirs(masked_dir, exist_ok=True)
images = os.listdir(rgb_dir)
for i in images:
    img = Image.open(os.path.join(rgb_dir, i))
    mask = Image.open(os.path.join(mask_dir, i))
    mask_np = np.array(mask)
    img_np = np.array(img)
    img_np[mask_np > 200] = 0
    Image.fromarray(img_np).save(os.path.join(masked_dir, i))
"""