import os
import numpy as np
from PIL import Image
import torch


embedding_vectors = torch.randn((209, 1792, 56, 56))
B, C, H, W = embedding_vectors.size()  # 550, 1792, 56, 56
embedding_vectors = embedding_vectors.view(B, C, H * W)

mean = torch.mean(embedding_vectors, dim=0).numpy() # 1792, 3136
print(f'mean : {mean.shape}')

cov = torch.zeros(C, C, H * W).numpy() # [1792, 1792, 3136]
print(f'cov : {cov.shape}')
