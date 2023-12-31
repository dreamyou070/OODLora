import torch
anormal_score_map = torch.randn((16, 256,1))
batch_num = 2
anormal_score_map_batch = torch.chunk(anormal_score_map, batch_num, dim=0) # batch, head, pixel_num, 1
for i in range(batch_num):
    a = anormal_score_map_batch[i]
    print(f'a : {a.shape}')