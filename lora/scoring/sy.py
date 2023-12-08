import torch
import numpy as np


#
"""
total_image_pred = np.array([])
batch, h,w = 3, 128,128
pred_mask = torch.randn((batch, 1, h,w))
out_mask = pred_mask
topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
image_score = torch.mean(topk_out_mask)
total_image_pred = np.append(total_image_pred, image_score.detach().cpu().numpy())
image_score = torch.mean(topk_out_mask)
total_image_pred = np.append(total_image_pred, image_score.detach().cpu().numpy())
print(total_image_pred)
"""
from PIL import Image
from sklearn.metrics import roc_auc_score,auc,average_precision_score
"""
orgin_latent = torch.randn((3,4,64,64))
orgin_latent = torch.flatten(orgin_latent, start_dim=1)

recon_latent = torch.randn((3,4,64,64))
recon_latent = torch.flatten(recon_latent, start_dim=1)

diff_latent = torch.nn.functional.mse_loss(orgin_latent, recon_latent, reduction='none')
diff_latent_np = diff_latent.detach().cpu().numpy() #* 0
trg_latent = np.zeros_like(diff_latent_np)

auroc_image = round(roc_auc_score(y_true = trg_latent,
                                  y_score = diff_latent_np,
                                  ), 3) * 100  # calculate score (not 0 ~ 1)
#print(f' (3.1) image score : {auroc_image}')
"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
roc_auc_score(y, clf.predict_proba(X)[:, 1])
#print(f'y : {y.shape}')
answer = clf.predict_proba(X)[:, 1]
#print(f'answer : {answer.shape}')
#0.99...
#>>> roc_auc_score(y, clf.decision_function(X))
#0.99...
"""
def load_64(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:,:]
    else:
        image = image_path
    h, w = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w= image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    new_pil = Image.fromarray(image).resize((64,64))
    image = np.array(new_pil)
    return image

answer_img_dir = '../examples/013_mask.png'
answer_np = load_64(answer_img_dir)
answer_np = np.where(answer_np > 100, 1, 0)
a = np.expand_dims(answer_np, axis=2)
a = np.repeat(a, 4, axis=-1)
print(f'answer_np : {a.shape}')
"""

diff_latent = torch.randn((3,4,64,64))
anomal_map = torch.sum(diff_latent, dim=1)
anomal_vector = torch.flatten(anomal_map, start_dim=1)
max_value = torch.max(anomal_vector, dim=1)[0].unsqueeze(1)
normalized_anomal_vector = anomal_vector / max_value
normalized_anomal_map = normalized_anomal_vector.view(3,1,64,64)
#
#print(f'diff_latent : {diff_latent.shape}')
import cv2
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

anomal_map = torch.randn((64,64)).detach().cpu()
gray = anomal_map * 255.0
print(f'ano_map : {gray}')
np_gray = np.uint8(gray)
print(f'np_gray : {np_gray}')
ano_map = cvt2heatmap(gray)
