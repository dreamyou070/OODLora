import torch


max_txt_idx = torch.tensor([[1,1],
                          [3,2]])
position_map = torch.where(max_txt_idx == 1, 1, 0) # [head, pixel_num]
print(f'position_map.shape : {position_map}')