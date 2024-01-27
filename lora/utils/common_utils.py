import os
import numpy as np
from PIL import Image
def get_lora_epoch(model_dir) :
    model_name = os.path.splitext(model_dir)[0]
    if 'last' not in model_name:
        model_epoch = int(model_name.split('-')[-1])
    else:
        model_epoch = 10000
    return model_epoch

def save_latent(latent, save_dir, h, w):
    print(f'save latent : {latent}')
    latent_np = latent.squeeze().detach().cpu().numpy().astype(np.uint8)
    latent_np = latent_np * 255
    print(f'save latent_np : {latent_np}')
    pil_img = Image.fromarray(latent_np).resize((h, w))
    pil_img.save(save_dir)
    return pil_img