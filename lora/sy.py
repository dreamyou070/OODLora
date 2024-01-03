import torch


trigger = torch.randn(64,64)
anormal_map = torch.flatten(trigger).unsqueeze(0)
print(f'anormal_map : {anormal_map.shape}')

print(f'attn : {attn.shape}')

