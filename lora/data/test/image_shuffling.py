import os
from PIL import Image
import numpy as np

org_img_dir = '000_recon.png'
org_pil = Image.open(org_img_dir).resize((512,512))
org_np = np.array(org_pil)
org_h, org_w, _ = org_np.shape
new_np = np.zeros((org_h, org_w, 3), dtype=np.uint8)
print(org_h // 2)
for i in range(org_h) :
    if i < org_h // 2:
        new_i = i + org_h // 2
    else:
        new_i = i - org_h // 2
    for j in range(org_w):
        new_np[new_i, j, :] = org_np[i, j, :]
shuffled_np = np.zeros((org_h, org_w, 3), dtype=np.uint8)
for i in range(org_w) :
    if i < org_w // 2:
        new_i = i + org_w // 2
    else:
        new_i = i - org_w // 2
    for j in range(org_h):
        shuffled_np[j, new_i, :] = new_np[j, i, :]


