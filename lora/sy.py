import os
import numpy as np
from PIL import Image
import torch, torchvision
import torch.nn.functional as nnf
import math
present_dir = os.path.dirname(__file__)
parent, _ = os.path.split(present_dir)

print("parent: ", parent)
print("present_dir: ", present_dir)