from PIL import Image
import numpy as np
import torch
mask_dir = '022.png'
mask_pil = Image.open(mask_dir).resize((32,32), Image.BICUBIC)
mask_img = np.array(mask_pil, np.uint8)
binary_img = np.where(mask_img > 10, 1, 0)
binary_img = torch.Tensor(binary_img)
for a in binary_img :
    print(a)