import os
import numpy as np
from PIL import Image
import torch
base_folder = r'for'
save_folder = r'mask'
os.makedirs(save_folder, exist_ok=True)
images = os.listdir(base_folder)
for image in images :
    img_dir = os.path.join(base_folder, image)
    pil_img = Image.open(img_dir).convert('RGB')
    masked_img = np.array(pil_img)
    binary_img = np.where(masked_img > 100, 1, 0)
    binary_img = torch.Tensor(binary_img)
    print(binary_img.shape)
    break
