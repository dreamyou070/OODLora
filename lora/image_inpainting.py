from diffusers import StableDiffusionInpaintPipeline
import torch, os
from PIL import Image

def main():
    print(f'\n step 1. make model')
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                          #revision="fp16",
                                                          #torch_dtype=torch.float16,
                                                          )
    prompt = "bagel"
    base_dir = '/data7/sooyeon/MyData/anomaly_detection/MVTecAD/bagel/test'
    classes = os.listdir(base_dir)
    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        image_dir = os.path.join(class_dir, 'rgb')
        mask_dir = os.path.join(class_dir, 'gt')
        inpainted_dir = os.path.join(class_dir, 'perfect_rgb')
        os.makedirs(inpainted_dir, exist_ok=True)
        images = os.listdir(image_dir)
        for i in images:
            image_path = os.path.join(image_dir, i)

            mask_path = os.path.join(mask_dir, i)
            image = pipe(prompt=prompt,
                         image=Image.open(image_path),
                         mask_image=Image.open(mask_path).convert('L'),).images[0]
            image.save(os.path.join(inpainted_dir, i))

if __name__ == "__main__":
    main()