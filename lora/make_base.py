import os
import numpy as np
from PIL import Image

a = np.ones((512,512)).astype(np.uint8)
a = a * 0
a = a.astype(np.uint8)
Image.fromarray(a).save('base.png')
