import torch
import numpy as np
from scipy.spatial.distance import mahalanobis

feat = torch.randn(320).unsqueeze(0)
features = [feat, feat]
normal_vectors = torch.cat(features, dim=0)  # sample, dim
print(normal_vectors.shape)
normal_vector_mean_torch = torch.mean(normal_vectors, dim=0)
normal_vectors_cov_torch = torch.cov(normal_vectors.transpose(0, 1))
print(f'normal_vector_mean_torch: {normal_vector_mean_torch.shape}')
print(f'normal_vectors_cov_torch: {normal_vectors_cov_torch.shape}')


def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

normal_vectors = torch.randn((10,30))
normal_vector_mean = torch.mean(normal_vectors, dim=0)

normal_vectors_mean = torch.mean(normal_vectors, dim=0).numpy()  # dim
normal_vectors_cov = np.cov(np.array(normal_vectors.cpu()), rowvar=False)
mahalanobis_dists = [mahalanobis(feat.cpu().numpy(), normal_vectors_mean, normal_vectors_cov) for feat in normal_vectors]


normal_vector_mean_torch = torch.mean(normal_vectors, dim=0)
normal_vectors_cov_torch = torch.cov(normal_vectors.transpose(0, 1))
mahalanobis_dists_2 = [mahalanobis(feat.detach().numpy(), normal_vector_mean_torch, normal_vectors_cov_torch) for feat in normal_vectors]
print(mahalanobis_dists_2)
def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

mahalanobis_dists = [mahal(feat, normal_vector_mean_torch, normal_vectors_cov_torch) for feat in normal_vectors]
dist_loss = torch.tensor(mahalanobis_dists)
print(dist_loss)