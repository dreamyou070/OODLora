import os
import numpy as np
from PIL import Image
import torch, torchvision
import torch.nn.functional as nnf
import math
masks = []
pix_num = 32*32
head = 8
mask = torch.zeros((8, pix_num))

inf = torch.tensor([980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740, 720,
        700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460, 440,
        420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160,
        140, 120, 100,  80,  60,  40,  20,   0])
inf = [999] + inf.tolist()
inference_times = torch.tensor(inf)
recon_1_times = inference_times[:1].tolist()

for i, t in enumerate(recon_1_times[:-1]):
        print(t)

