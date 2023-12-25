import os
import numpy as np
from PIL import Image
a = np.zeros((512,512)).astype(np.uint8)
a[0,0] = 1
a = a * 100
Image.fromarray(a).show()
Image.fromarray(a).save('base.png')