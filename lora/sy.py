import os
import numpy as np
from PIL import Image
import torch


embedding_vectors = torch.randn((3,4,64,64))
B, C, H, W = embedding_vectors.size()
embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()  # [N, 550, 3136]
img_level_dist = []
for i in range(H * W):
    pixel_level_dist = []
    for sample in embedding_vectors: # for B number
        m_dist = np.array(1.3)
        pixel_level_dist.append(m_dist)
    img_level_dist.append(pixel_level_dist)
score_map = np.array(img_level_dist).transpose(1, 0).reshape(B, H, W)
model_score = np.mean(score_map, axis=0)
model_score = np.sum(model_score)
print(model_score)
