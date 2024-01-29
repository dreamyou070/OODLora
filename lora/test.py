import torch
anomal_features = []
feat = torch.randn(10)
anomal_features.append(feat)
anomal_features.append(feat)

anomal_features = torch.stack(anomal_features, dim=0)
print(anomal_features.shape)
mean_vector = torch.mean(anomal_features, dim=0)
print(mean_vector.shape)