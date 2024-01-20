import os
from PIl import Image
base_file = r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/cookie/test/crack/gt/000.png'
h,w = Image.open(base_file).size

print(h)