import torch
from PIL import Image


def get_dino_dim(image_dir, dino_model, dino_transform, args):
    pil_img = Image.open(image_dir)  # RGB
    pil_img = pil_img.convert('RGB')
    tensor_image = dino_transform(pil_img).unsqueeze(0)  # 1, 3, h, w
    tensor_image = tensor_image.to(args.device)
    image_embedding = dino_model.get_embeddings(tensor_image)
    return image_embedding


class VITs16():
    def __init__(self, dino_model, device="cuda"):
        self.model = torch.hub.load('facebookresearch/dino:main', dino_model).to(device)
        #self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
        self.model.eval()

    def get_embeddings(self, tensor_image):
        output = self.model(tensor_image)
        return output

    def get_embeddings_intermediate(self, tensor_image, n_last_block=4):
        """
        We use `n_last_block=4` when evaluating ViT-Small
        """
        intermediate_output = self.model.get_intermediate_layers(tensor_image, n=n_last_block)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        return output


