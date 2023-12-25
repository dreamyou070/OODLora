import os
import numpy as np
from PIL import Image
a = np.zeros((512,512)).astype(np.uint8)
a[0:50,0:50] = 1
a = a * 255
Image.fromarray(a).show()
Image.fromarray(a).save('base.png')