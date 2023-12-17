import torch
filling_label = 0.0
label_tensor = torch.tensor(1).fill_(filling_label)
print(label_tensor)