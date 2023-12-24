import numpy as np
import random
import torch

"""
alphas_cumprod = scheduler.alphas_cumprod
sqrt_alpha_prod = alphas_cumprod[700] ** 0.5
"""
a = torch.IntTensor([700])
a.to('cpu')