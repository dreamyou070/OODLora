import torch
from sklearn.decomposition import PCA
import numpy as np
# -------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=0)
d = 10
features = torch.randn((4,d))
features = np.array(features)
feat_ = pca.fit_transform(features)
mean = np.mean(feat_, axis=0)
print(mean)





"""
normal_vectors = []

pdist = torch.nn.PairwiseDistance(p=2)
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
"""