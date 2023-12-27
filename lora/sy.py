import os
import numpy as np
from PIL import Image
import torch


from random import sample
mask_img = Image.fromarray(np.zeros((512,512)).astype(np.uint8))
mask_img.show()