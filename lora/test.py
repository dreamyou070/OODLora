import os
from PIL import Image
import numpy as np

base_img_dir = '/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/peach/test/cut/gt/000.png'
h,w = Image.open(base_img_dir).size

base_folder = r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/peach/test/good/rgb'
gt_folder = r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/peach/test/good/gt'
images = os.listdir(base_folder)
for image in images:
    base = np.zeros((h,w))
    mask = Image.fromarray(base.astype(np.uint8))
    mask.save(os.path.join(gt_folder, image))