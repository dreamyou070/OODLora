import os
from PIL import Image
import numpy as np

base_folder = r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/cookie/test/good/rgb'
gt_folder = r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/cookie/test/good/gt'
images = os.listdir(base_folder)
for image in images:
    base = np.zeros((500,500))
    mask = Image.fromarray(base.astype(np.uint8))
    mask.save(os.path.join(gt_folder, image))