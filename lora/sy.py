import os
import numpy as np
from PIL import Image
import torch


feat1 = torch.randn((209, 256, 56, 56))
feat2 = torch.randn((209, 512, 28, 28))
feat3 = torch.randn((209, 1024, 14, 14))

feat1 = torch.cat([feat1], 0)
#feat2 = torch.cat((fea2), dim=0)
#feat3 = torch.cat((fea3), dim=0)
print(feat1.shape)