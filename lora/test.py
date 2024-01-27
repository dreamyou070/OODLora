import torch
normal_vectors = []
n_vectors_1 = torch.rand((3,4))
n_vectors_2 = torch.rand((5,4))
normal_vectors.append(n_vectors_1)
normal_vectors.append(n_vectors_2)
n_vectors = torch.cat(normal_vectors, dim=0)
n_center = n_vectors.mean(dim=0)
print(n_center)