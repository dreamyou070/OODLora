import torch

flatten_img_mask = torch.randn((4,2))
position_map = torch.tensor([[0,0],
                             [1,0],
                             [0,1],
                             [0,1]])
score_pairs = []
anormal_pos = []
for i in range(flatten_img_mask.shape[0]):
    position_info = position_map[i]
    if position_info[0] == 1 or position_info[1] == 1:
        score_pair = flatten_img_mask[i]
        anormal_pos.append(position_info[1])
        score_pairs.append(score_pair)
score_pairs = torch.stack(score_pairs)
anormal_pos = torch.stack(anormal_pos)
print(score_pairs.shape)
print(anormal_pos.shape)
