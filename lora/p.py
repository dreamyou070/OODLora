import torch
attention_probs = torch.randn((8, 4096, 4096))
attention_probs_1 = torch.sum(attention_probs, dim=0)
attention_probs_2 = torch.sum(attention_probs, dim=0)
a = [attention_probs_1, attention_probs_2]
a = torch.stack(a, dim=0)
prob_map = torch.mean(a, dim=0)
height = 64
pix_num = height ** 2
diagonal_mask = torch.eye(pix_num)
normal_score = prob_map * (diagonal_mask)
normal_score_vector = torch.sum(normal_score, dim=-1)
normal_map = normal_score_vector.reshape(height, height)
max_value = torch.max(normal_map)
normal_map = normal_map/max_value
anormal_map = 1 - normal_map