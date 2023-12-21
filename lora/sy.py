from torch.nn import L1Loss, MSELoss
import torch
import collections
alphas_cumprod = torch.tensor([0.0105, 0.0104, 0.0103, 0.0102, 0.0101, 0.0100, 0.0098, 0.0097,
        0.0096, 0.0095, 0.0094, 0.0093, 0.0092, 0.0091, 0.0090, 0.0089, 0.0088,
        0.0087, 0.0086, 0.0085, 0.0084, 0.0083, 0.0082, 0.0082, 0.0081, 0.0080,
        0.0079, 0.0078, 0.0077, 0.0076, 0.0075, 0.0074, 0.0074, 0.0073, 0.0072,
        0.0071, 0.0070, 0.0070, 0.0069, 0.0068, 0.0067, 0.0066, 0.0066, 0.0065,
        0.0064, 0.0063, 0.0063, 0.0062, 0.0061, 0.0061, 0.0060, 0.0059, 0.0058,
        0.0058, 0.0057, 0.0056, 0.0056, 0.0055, 0.0054, 0.0054, 0.0053, 0.0053,
        0.0052, 0.0051, 0.0051, 0.0050, 0.0049, 0.0049, 0.0048, 0.0048, 0.0047,
        0.0047])
timesteps = torch.tensor([1,2,3])
next_timesteps = (timesteps + 1).tolist()
alpha_prod_t_next = alphas_cumprod[next_timesteps]
alpha_prod_t = alphas_cumprod[timesteps.tolist()]
print(alpha_prod_t_next)
print(alpha_prod_t)


#
beta = (1 - alpha_prod_t_next)**0.5
gamma = ((alpha_prod_t_next/alpha_prod_t) * (1-alpha_prod_t)) ** 0.5
print(beta)
print(gamma)

diff = torch.randn((3,4,64,64))
beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
beta = beta.expand(diff.shape)
print(beta[0,:,:,:])
