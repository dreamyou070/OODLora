from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

def main() :
    #url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    #image = Image.open(requests.get(url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8',
                                                  cache_dir='/home/dreamyou070/pretrained_models')
    model = ViTModel.from_pretrained('facebook/dino-vitb8',
                                     cache_dir='/home/dreamyou070/pretrained_models')

    #inputs = processor(images=image, return_tensors="pt")
    #outputs = model(**inputs)
    #last_hidden_states = outputs.last_hidden_state

if __name__ == '__main__' :
    main()