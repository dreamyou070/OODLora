import os
import torch

mu = torch.randn(320)
cov = torch.randn((320, 320))


from torch.distributions.multivariate_normal import MultivariateNormal
m = MultivariateNormal(mu, torch.eye(320))
random_vector = m.sample()
print(random_vector.shape)