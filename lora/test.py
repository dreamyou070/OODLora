import torch
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

featurs = torch.tensor([[1, 2, 3],
                        [1, 5, 6],
                        [1, 8, 9],
                        [1, 0, 9]]).to(torch.float32)
normal_vectors_mean = torch.mean(featurs, dim=0).numpy()  # dim
normal_vectors_cov = np.cov(featurs.detach().cpu().numpy(), rowvar=False)
normal_vectors_cov_tensor = torch.tensor(normal_vectors_cov)
dim = 3
identity_matrix = torch.eye(dim)
variance_vector = (normal_vectors_cov_tensor * identity_matrix).sum(dim=1).squeeze()
sort_result, index_result = torch.sort(variance_vector, descending=False)
max_dim = 2
print(sort_result)
important_dim = index_result[:max_dim]
important_dim, _ = torch.sort(important_dim)
print(f'before sorting, important_dim: {important_dim}')

print(f'after sorting, important_dim: {important_dim}')

redused_features = []
for i in important_dim :
    i_values = featurs[:,i]
    redused_features.append(i_values)
redused_features = torch.stack(redused_features, dim=1)
print(redused_features)
