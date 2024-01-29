import torch

real_latent = torch.randn(1,4,64,64)
mean = torch.mean(real_latent, dim=1)
std = torch.std(real_latent, dim=1)
print(mean.shape)