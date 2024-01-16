from PIL import Image
import numpy as np

rgb_dir = 'rgb_combined_014.png'
rgb_channel = Image.open(rgb_dir).resize((512,512))
np_rgb = np.array(rgb_channel)

mask_dir = 'sam_combined_014.png'
alpha_channel = Image.open(mask_dir).resize((512,512))
alpha_channel_np = np.array(alpha_channel) / 255


base_np = np.ones((512,512,4))
base_np[:,:,:3] = np_rgb
base_np[:,:,-1] = alpha_channel_np
rgb_alpha = Image.fromarray(base_np.atype(np.uint8))
rgb_alpha.show()
