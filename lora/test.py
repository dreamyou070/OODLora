import queue
import torch

random_anomal_positions = torch.tensor([1,2,3])
head_num = 8
anormal_mask = torch.stack([random_anomal_positions for i in range(head_num)],dim=0)  # .unsqueeze(-1)  # 8, res*res
print(anormal_mask.shape)