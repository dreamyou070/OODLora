import os
import numpy as np
from PIL import Image
import torch, torchvision
import torch.nn.functional as nnf
import math

inference_times = torch.tensor([999, 980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740,
        720, 700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460,
        440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180,
        160, 140, 120, 100,  80,  60,  40,  20,   0]).tolist()
index = inference_times.index(300)
recon_times = inference_times[index:]
print(index)
print(recon_times)
