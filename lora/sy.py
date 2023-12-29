import torch

text_encoder_conds = torch.randn(8,77,768)
text_encoder_cond = text_encoder_conds[:, :2, :]
print(text_encoder_cond.shape)