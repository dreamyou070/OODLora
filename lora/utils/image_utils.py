from PIL import Image
import numpy as np
from torchvision import transforms
import torch

IMAGE_TRANSFORMS = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.5], [0.5]),])
def load_image(image_path, trg_h, trg_w):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    if trg_h and trg_w :
        image = image.resize((trg_w, trg_h), Image.BICUBIC)
    img = np.array(image, np.uint8)
    return img

@torch.no_grad()
def image2latent(image, vae, device, weight_dtype):
    if type(image) is Image:
        image = np.array(image)
    if type(image) is torch.Tensor and image.dim() == 4:
        latents = image
    else:
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device, weight_dtype)
        latents = vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
    return latents

@torch.no_grad()
def latent2image(latents, vae, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    return image