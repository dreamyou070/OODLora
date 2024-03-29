import os
from PIL import Image
import numpy as np

class_name = 'tire'
base_img_dir = f'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/{class_name}/test/cut/gt/000.png'
w,h = Image.open(base_img_dir).size

base_folder = f'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/{class_name}/test/good/rgb'
gt_folder = f'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/{class_name}/test/good/gt'
images = os.listdir(base_folder)
for image in images:
    base = np.zeros((h,w))
    mask = Image.fromarray(base.astype(np.uint8))
    mask.save(os.path.join(gt_folder, image))