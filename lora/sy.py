import torch


score_diff = torch.randn((8, 32*32,1))
score_list = [score_diff,score_diff,score_diff,score_diff,score_diff]
score_diff = torch.cat(score_list, dim=0).mean(dim=0).squeeze()
print(f'score_diff.shape (32*32) : {score_diff.shape}')