import torch
import numpy as np
from sklearn.decomposition import PCA

query_1 = torch.randn((4,5))
query_2 = torch.randn((4,5))
query = torch.concat((query_1, query_2), dim=-1)
print(f'query : {query.shape}')