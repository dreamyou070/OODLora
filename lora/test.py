import queue
import torch

random_anomal_positions = torch.tensor([1,2,3])
head_num = 8
random_anomal_positions = torch.randn(4096)
random_anormal_position = torch.stack([random_anomal_positions.unsqueeze(0) for i in range(head_num)], dim=0)
print(random_anormal_position.shape)