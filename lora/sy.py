import torch, ast

embedding = torch.randn(1,77,768)
my_embedding = torch.randn(1,16, 768)

cls_embedding = embedding[:,0,:].unsqueeze(0)
other_embedding = embedding[:,2:,:]
total_embedding = torch.cat((cls_embedding, my_embedding, other_embedding), dim=1)
print(total_embedding.shape)

other_embeddding_ = total_embedding[:,17:,:]
print(other_embeddding_.shape)