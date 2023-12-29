import torch


attention_probs = torch.randn((8, 32*32,76))
index_info = attention_probs.max(dim=-1).indices
position_map = torch.where(index_info==0, 1, 0)
print(position_map.shape)