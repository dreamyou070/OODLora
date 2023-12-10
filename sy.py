from PIL import Image
import numpy as np
mask_dir = 'example/test_dataset/gt/000.png'
mask_img = Image.open(mask_dir)
mask_img = mask_img.convert("L").resize((512, 512))
np_img = np.array(mask_img)
np_img = np.where(np_img > 200, 255, 0)
mask_img = Image.fromarray(np_img).convert("RGB")
mask_img.show()
"""
np_img = np.array(mask_img.resize((512, 512)))
# 
torch_img = torch.from_numpy(np_img)
mask_img = torch_img / 255.0  # 0~1
mask_imgs.append(mask_img)
"""
from diffusers.image_processors import VaeImageProcessor
vae_scale_factor =
mask_processor = VaeImageProcessor( vae_scale_factor=vae_scale_factor,
                                    do_normalize=False, do_binarize=True, do_convert_grayscale=True)
heigh = 512
width = 512
mask_condition = mask_processor.preprocess(mask_image,
                                           height=height,
                                           width=width)