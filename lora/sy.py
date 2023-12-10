from PIL import Image
import numpy as np
mask_dir = '../example/test_dataset/gt/000.png'
mask_img = Image.open(mask_dir)
mask_img = mask_img.convert("L").resize((512, 512))
np_img = np.array(mask_img)
np_img = np.where(np_img > 200, 255, 0)
#mask_img = Image.fromarray(np_img).convert("RGB")

"""
np_img = np.array(mask_img.resize((512, 512)))
# 
torch_img = torch.from_numpy(np_img)
mask_img = torch_img / 255.0  # 0~1
mask_imgs.append(mask_img)
"""
#from diffusers.image_processor import VaeImageProcessor
vae_scale_factor = 0.18215
#mask_processor = VaeImageProcessor( vae_scale_factor=vae_scale_factor,
#                                    do_normalize=False, do_binarize=True, do_convert_grayscale=True)
heigh = 512
width = 512
#mask_condition = mask_processor.preprocess(mask_image,
#                                           height=height,
#                                           width=width)

import torch
def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

image = [np_img,np_img]
image =np.stack(image, axis=0)
image =numpy_to_pt(image)
height, width = 512, 512
image = resize(image, height, width)

#image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
#print(image.shape)
            #image = self.numpy_to_pt(image)
            #height, width = self.get_default_height_width(image, height, width)
            #if self.config.do_resize:
             #   image = self.resize(image, height, width)
