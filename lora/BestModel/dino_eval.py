from transformers import ViTImageProcessor, ViTModel
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import numpy as np
import torch.nn as nn


cos = nn.CosineSimilarity(dim=0)

def main() :

    #url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    #image = Image.open(requests.get(url, stream=True).raw)

    #processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8',cache_dir='../../../../pretrained_models')
    #model = ViTModel.from_pretrained('facebook/dino-vitb8',cache_dir='../../../../pretrained_models')
    device = 'cuda:0'
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base',cache_dir='../../../../pretrained_models')
    model = AutoModel.from_pretrained('facebook/dinov2-base',cache_dir='../../../../pretrained_models').to(device)

    img_base_folder = '../../../../sample'
    lora_epochs = os.listdir(img_base_folder)
    for lora_epoch in lora_epochs:
        lora_epoch_folder = os.path.join(img_base_folder, lora_epoch)
        train_datasets = os.path.join(lora_epoch_folder, 'train_dataset')
        test_datasets = os.path.join(lora_epoch_folder, 'test_dataset')
        cats = os.listdir(train_datasets)
        for cat in cats:
            train_cat_dir = os.path.join(train_datasets, cat)
            pure_names = os.listdir(train_cat_dir)
            for pure_name in pure_names:
                pure_name_dir = os.path.join(train_cat_dir, pure_name)

                pure_img_dir = os.path.join(pure_name_dir, f'{pure_name}.png')
                mask_img_dir = os.path.join(pure_name_dir, f'{pure_name}_mask.png')
                recon_img_dir = os.path.join(pure_name_dir, f'{pure_name}_recon_0.png')

                pure_img = Image.open(pure_img_dir).resize((512,512))
                pure_np_img = np.array(pure_img)
                recon_img = Image.open(recon_img_dir).resize((512,512))
                recon_np_img = np.array(recon_img)
                mask_img = Image.open(mask_img_dir).resize((512,512)).convert('RGB')
                mask_np_img = np.array(mask_img)
                mask_np_img = np.where(mask_np_img > 100, 1, 0)

                total_area = np.ones(mask_np_img.shape).sum()

                background_1 = (pure_np_img * (1-mask_np_img)).astype(np.uint8)
                background_2 = (recon_np_img * (1-mask_np_img)).astype(np.uint8)

                object_1 = (pure_np_img * mask_np_img).astype(np.uint8)
                object_2 = (recon_np_img * mask_np_img).astype(np.uint8)

                background_1 = Image.fromarray(background_1 )
                background_2 = Image.fromarray(background_2)
                object_1 = Image.fromarray(object_1)
                object_2 = Image.fromarray(object_2)

                background_1_inputs = processor(background_1, return_tensors="pt").to(device)
                background_1_outputs = model(**background_1_inputs).last_hidden_state.mean(dim=1)
                background_2_inputs = processor(background_2, return_tensors="pt").to(device)
                background_2_outputs = model(**background_2_inputs).last_hidden_state.mean(dim=1)
                back_sim = cos(background_1_outputs[0], background_2_outputs[0]).item()
                back_area = (1-mask_np_img).sum()
                back_sim = back_sim * (back_area/total_area)





                object_1_inputs = processor(object_1, return_tensors="pt").to(device)
                object_1_outputs = model(**object_1_inputs).last_hidden_state.mean(dim=1)
                object_2_inputs = processor(object_2, return_tensors="pt").to(device)
                object_2_outputs = model(**object_2_inputs).last_hidden_state.mean(dim=1)
                obj_sim = cos(object_1_outputs[0], object_2_outputs[0]).item()
                obj_area = mask_np_img.sum()
                obj_sim = obj_sim * (obj_area/total_area)
                print(f'{pure_name} : back sim = {back_sim}, obj sim = {obj_sim}')


                break
            break
        break



    #inputs = processor(images=image, return_tensors="pt")
    #outputs = model(**inputs)
    #last_hidden_states = outputs.last_hidden_state

if __name__ == '__main__' :
    main()