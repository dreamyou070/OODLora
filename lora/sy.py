import torch

attention_scores = torch.randn((8, 64, 77))
attention_probs = attention_scores.softmax(dim=-1)
max_txt_idx = torch.max(attention_probs, dim=-1).indices
map = torch.where(max_txt_idx == 1, 1, 0)
print(map)