import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
"""
base_dir = '/home/dreamyou070/Lora/OODLora/result/MVTec3D-AD_experiment/bagel/lora_training/normal/random_vector_generating'

mu_dir =  os.path.join(base_dir, f'record_lora_eopch_6_mahalanobis/normal_vector_mean_torch.pt')
cov_dir = os.path.join(base_dir, f'record_lora_eopch_6_mahalanobis/normal_vectors_cov_torch.pt')

mu = torch.load(mu_dir)
cov = torch.load(cov_dir)

def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

random_vector_generator = MultivariateNormal(mu, torch.eye(320))
for i in range(10) :
    random_vector = random_vector_generator.sample()
    dist = mahal(random_vector, mu, cov)
    print(f'{i} trial dist = {dist}')
"""

normal_vectors = torch.randn((10,320))
normal_vectors_np = normal_vectors.numpy()
normal_vectors_np_t = normal_vectors_np.T

mu = torch.mean(normal_vectors, dim=0)
cov = torch.cov(normal_vectors.transpose(0, 1))
cov_np_1 = np.cov(normal_vectors_np)
cov_np_2 = np.cov(normal_vectors_np_t)
#print(f'normal_vectors = {normal_vectors.shape}')
print(f'normal_vectors_np = {normal_vectors_np.shape}')
print(f'normal_vectors_np_t = {normal_vectors_np_t.shape}')

print(f'cov_np_1 = {cov_np_1.shape}')
print(f'cov_np_2 = {cov_np_2.shape}')
#from torch.distributions.multivariate_normal import MultivariateNormal
h = np.array([170, 188, 165, 176, 160, 181, 178])
w = np.array([65, 82, 58, 68, 50, 71, 70])
cov_hw = np.cov(h, w, ddof = 1)


#random_vector_generator = MultivariateNormal(loc = mu,
#                                             covariance_matrix =cov)
from scipy.spatial.distance import mahalanobis
mean = torch.randn(320)
conv_inv = torch.eye(320)
sample = torch.randn(320)


embedding_vectors = torch.randn(10, 320)
cov = np.cov(embedding_vectors.numpy(), rowvar=False)

cov_torch = torch.tensor(cov)
cov_torch = torch.eye(320) * cov_torch
print(f'cov = {cov_torch}')
dist = mahalanobis(sample, mean, cov_torch)
print(f'dist = {dist}')
random_vector_generator = MultivariateNormal(mean, cov_torch)
random_vector = random_vector_generator.sample()
dist = mahalanobis(random_vector, mean, cov_torch)
print(f'dist = {dist}')
