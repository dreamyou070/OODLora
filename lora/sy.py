import os
import numpy as np
from PIL import Image
import torch

#   diff = torch.sum(diff, axis=1)
#print(diff)
image = torch.randn((8,64))
image = 255 * image / image.max()
print(image.shape)
image = image.unsqueeze(-1).expand(*image.shape, 3)
print(image.shape)
image = image.numpy().astype(np.uint8)
pil = Image.fromarray(image).resize((256, 256))
image = np.array(pil)
print(image.shape)
#
  #      image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
   #     images.append(image)
"""
image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
"""