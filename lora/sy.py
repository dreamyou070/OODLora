import os
import numpy as np
from PIL import Image
import torch


map = torch.randn(8,1024)
map = map.mean(dim=0)