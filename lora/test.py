import torch

query1 = torch.randn(4,10)
query2 = torch.randn(4,10)

query_matrix = torch.matmul(query1, query2.t()) # 4 * 4
query_matrix = torch.softmax(query_matrix, dim=1)
print(query_matrix)
pix_num = query_matrix.size(0)
identity_matrix = torch.eye(pix_num)
similarity_vector = query_matrix.diag() # 4 * 1
print(similarity_vector)
min_score = similarity_vector.min()
if min_score < 0:
    similarity_vector = similarity_vector + min_score
max_score = similarity_vector.max()
similarity_vector = similarity_vector / max_score
print(similarity_vector)