import torch

normal_trigger_activation = torch.randn(8, 64*64)
normal_trigger_activation = normal_trigger_activation.mean(dim=0)
print(normal_trigger_activation.shape)