import torch

org_query = torch.randn(4096).unsqueeze(-1)
recon_query = torch.randn(4096).unsqueeze(-1)
anomaly_score = (org_query @ recon_query.T).cpu()
pix_num = anomaly_score.shape[0]
res = int(pix_num ** 0.5)
anomaly_score = (torch.eye(pix_num) * anomaly_score).sum(dim=0)
print(anomaly_score)