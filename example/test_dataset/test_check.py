import os
from PIL import Image
import numpy as np

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