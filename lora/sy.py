import os
import numpy as np
from PIL import Image
import torch


res = (512,512)
res = str(res)
res = res.replace('(','')
res = res.replace(')','')
print(res)
print(type(res))