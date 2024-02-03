import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

base_dir = '/home/dreamyou070/Lora/OODLora/result/MVTec3D-AD_experiment/bagel/lora_training/normal/random_vector_generating/'

mu_dir =  os.path.join(base_dir, f'record_lora_eopch_7_mahalanobis_only_normal/normal_vector_mean_torch.pt')
cov_dir = os.path.join(base_dir, f'record_lora_eopch_7_mahalanobis_only_normal/normal_vectors_cov_torch.pt')

mu = torch.load(mu_dir)
cov = torch.load(cov_dir)
cov = cov * torch.eye(320)

def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

random_vector_generator = MultivariateNormal(mu, cov)
for i in range(100) :
    random_vector = random_vector_generator.sample()
    dist = mahal(random_vector, mu, cov)
    print(f'{i} trial dist = {dist}')
