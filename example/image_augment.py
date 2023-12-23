import os
from PIL import Image
import numpy as np
import random
base_dir = '000.png'
img = Image.open(base_dir)
np_img = np.array(img)
org_h, org_w = img.size
patch_h, patch_w = org_h/4, org_w/4
h_num, w_num = 4, 4
total_patch_num = h_num*w_num
patch_indexs = [i for i in range(total_patch_num)]
random.shuffle(patch_indexs)

patches = []
for i in range(h_num):
    for j in range(w_num):
        patch = np_img[int(i*patch_h):int((i+1)*patch_h), int(j*patch_w):int((j+1)*patch_w)]
        patches.append(patch)
zero_image = np.zeros((org_h, org_w, 3), dtype=np.uint8)

for i in range(h_num):
    for j in range(w_num):
        p_index = patch_indexs.pop()
        patch = patches[p_index]
        zero_image[int(i*patch_h):int((i+1)*patch_h), int(j*patch_w):int((j+1)*patch_w)] = patch
shuffled_img = Image.fromarray(zero_image)
shuffled_img.show()