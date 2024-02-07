import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
"""
network_weights = '/home/dreamyou070/Lora/OODLora/result/MVTec3D-AD_experiment/bagel/lora_training/normal/res_64_down_1_task_loss_mahal_dist_attn_loss_0.008_actdeact_mahal_anomal/models/epoch-000003.safetensors'
base, model_name = os.path.split(network_weights)
name, ext = os.path.splitext(model_name)
model_epoch_int = int(name.split('-')[-1])
base_dir, _ = os.path.split(base)

mu_dir =  os.path.join(base_dir, f'record_lora_eopch_{model_epoch_int}_mahalanobis/normal_vector_mean_torch.pt')
cov_dir = os.path.join(base_dir, f'record_lora_eopch_{model_epoch_int}_mahalanobis/normal_vectors_cov_torch.pt')
"""
mu = torch.randn(320)
cov = torch.eye(320)
random_vector_generator = MultivariateNormal(mu, cov)
#random_vector_generator.to('cuda')
def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)
feat= torch.randn(320)
nomal_features = [feat.unsqueeze(0) for _ in range(10)]
normal_vectors = torch.cat(nomal_features, dim=0)  # sample, dim
print(normal_vectors.shape)