import os
import numpy as np
from PIL import Image
import torch

diff = [1,2,3]

diff_score = np.mean(np.array(diff))
print(diff_score)