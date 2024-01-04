import torch, ast

cls_embedding = torch.randn(1,1,768)
other_embedding = torch.randn(1,16,768)
embedding = torch.cat((cls_embedding, other_embedding), dim=1)
print(embedding.shape)