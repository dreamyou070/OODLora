from torchvision import transforms
from PIL import Image
import torch
import torchvision
attn_transforms = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                                          transforms.ToTensor(),])

img_mask_dir = '000.png'
mask_img = Image.open(img_mask_dir).convert('L').resize((512, 512), Image.BICUBIC)
mask_img = attn_transforms(mask_img)
mask_list = [mask_img]
mask_img = torch.stack(mask_list, dim=0)
resize_transform = transforms.Resize((32,32),)
resized_mask = resize_transform(mask_img)
img_masks_res = (resized_mask == 0.0).float() # background = 0, foreground = 1
print('img_masks_res.shape (1,1,32,32) : ', img_masks_res.shape)
transforms.ToTensor()(Image.open(img_mask_dir).convert('L').resize((32, 32), Image.BICUBIC))