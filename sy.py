import random, os
import numpy as np
from PIL import Image
folder = 'rgb'
images = os.listdir(folder)
for image in images:
    img_dir = os.path.join(folder, image)
    np_img = np.array(Image.open(img_dir))
    h, w, c = np_img.shape
    trg_h_1, trg_w_1 = h / 3, w / 3
    trg_h_2, trg_w_2 = h * (2/3), w * (2/3)
    positions = [[trg_h_1, trg_w_1],[trg_h_2, trg_w_2]]
    labels = [1,1]
