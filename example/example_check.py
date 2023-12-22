import os
from PIL import Image
import numpy as np
folder = 'ground_truth'
images = os.listdir(folder)
for image in images:
    image_dir = os.path.join(folder, image)
    image = Image.open(image_dir)
    np_image = np.array(image)
    np.where()
    np_image = np_image.flatten()
    image_list = np_image.tolist()
    set_img = set(image_list)
    print(f'{image_dir} : {set_img}')
