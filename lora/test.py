import queue
import torch


def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)


normal_vector_good_score = torch.randn((3,10))
normal_vector_mean_torch = torch.mean(normal_vector_good_score, dim=0)
normal_vectors_cov_torch = torch.cov(normal_vector_good_score.transpose(0, 1))
# ----------------------------------------------------------------------------------------------------------- #
# [1] good mahalanobis distances
mahalanobis_dists = [mahal(feat, normal_vector_mean_torch, normal_vectors_cov_torch) for
                     feat in normal_vector_good_score]
import matplotlib.pyplot as plt

plt.figure()
plt.hist(mahalanobis_dists)
plt.show()