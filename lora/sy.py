import torch


trigger = torch.randn(32,32)
anormal_map = torch.flatten(trigger).unsqueeze(0)
anormal_map = anormal_map.repeat(8, 1)
print(anormal_map.shape)