import torch
normal_vectors = []

pdist = torch.nn.PairwiseDistance(p=2)
d = 2
features = torch.randn((4,d))
pix_num = features.shape[0]
n_vector = torch.randn((d)).unsqueeze(0).repeat(pix_num, 1)
b_vectpr = torch.randn((d)).unsqueeze(0).repeat(pix_num, 1)
n_diff = pdist(features, n_vector)
b_diff = pdist(features, b_vectpr)
total_diff = n_diff + b_diff
n_diff = n_diff / total_diff
b_diff = b_diff / total_diff
diff = torch.where(n_diff > b_diff, b_diff, n_diff)
diff = diff / diff.max()
mask = 1-diff
print(f'features: {features}')
print(f'n_vector: {n_vector}')
print(diff)