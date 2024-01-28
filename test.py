import torch
from random import sample
import numpy as np

query = torch.randn((1,4,64,64))
b, c, h, w = query.shape

for i in range(h) :
    original_feature = query[:,:,i,1].squeeze()
    shuffle = torch.randperm(c)
    new_feature = original_feature[shuffle]
    query[:,:,i,1] = new_feature
