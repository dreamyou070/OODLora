import torch
import numpy as np
from scipy.spatial.distance import mahalanobis

normal_vectors = torch.randn((10,30))
normal_vector_mean = torch.mean(normal_vectors, dim=0)

normal_vectors_mean = torch.mean(normal_vectors, dim=0).numpy()  # dim
normal_vectors_cov = np.cov(np.array(normal_vectors.cpu()), rowvar=False)
mahalanobis_dists = [mahalanobis(feat.cpu().numpy(), normal_vectors_mean, normal_vectors_cov) for feat in normal_vectors]


normal_vector_mean_torch = torch.mean(normal_vectors, dim=0)
normal_vectors_cov_torch = torch.cov(normal_vectors.transpose(0, 1))
mahalanobis_dists = [mahalanobis(feat.detach().numpy(), normal_vector_mean_torch, normal_vectors_cov_torch) for feat in normal_vectors]