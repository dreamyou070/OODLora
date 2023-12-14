from __future__ import annotations
from typing import Sequence
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image


IMAGE_TRANSFORMS = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.5], [0.5]),])


class SYDataset(Dataset):

    def __init__(self, image_dir, masked_dir, h, w, class_caption,tokenizers,
                 tokenizer_max_length) -> None:

        self.image_dir = image_dir
        self.masked_dir = masked_dir
        self.w, self.h = w, h
        self.data = []
        image_classes = os.listdir(self.image_dir)
        for image_class in image_classes :
            repeat, caption = image_class.split('_')
            image_class_dir = os.path.join(self.image_dir, image_class)
            mask_class_dir = os.path.join(self.masked_dir, image_class)
            image_names = os.listdir(image_class_dir)
            for image_name in image_names :
                image_path = os.path.join(image_class_dir, image_name)
                mask_path = os.path.join(mask_class_dir, image_name)
                self.data.append({'image': image_path,
                                  'masked': mask_path,
                                  'caption' : caption,
                                  'class_caption' : class_caption})

        self.tokenizers = tokenizers
        self.tokenizer_max_length =tokenizer_max_length

    def __len__(self) -> int:
        return len(self.data)

    def get_input_ids(self, caption, tokenizer=None):

        if tokenizer is None:
            tokenizer = self.tokenizers[0]

        tokenizer_output = tokenizer(caption, padding="max_length", truncation=True,
                                     max_length=self.tokenizer_max_length, return_tensors="pt")

        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask

        if self.tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                for i in range( 1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2) :
                    ids_chunk = (input_ids[0].unsqueeze(0),input_ids[i : i + tokenizer.model_max_length - 2],  input_ids[-1].unsqueeze(0),)
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                for i in range(1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id
                    iids_list.append(ids_chunk)
            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids, attention_mask

    def generate_text_embedding(self, caption, class_caption, tokenizer):
        cls_token = 49406
        pad_token = 49407
        token_input = tokenizer([class_caption], padding="max_length", max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt", )  # token_input = 24215
        token_ids = token_input.input_ids[0]
        token_attns = token_input.attention_mask[0]
        trg_token_id = []
        for token_id, token_attn in zip(token_ids, token_attns):
            if token_id != cls_token and token_id != pad_token and token_attn == 1:
                # token_id = 24215
                trg_token_id.append(token_id)
        text_input = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length,
                               truncation=True, return_tensors="pt", )
        token_ids = text_input.input_ids
        attns = text_input.attention_mask
        for token_id, attn in zip(token_ids, attns):
            trg_indexs = []
            for i, id in enumerate(token_id):
                if id in trg_token_id:
                    trg_indexs.append(i)
        return trg_indexs

    def load_image(self, image_path, trg_h, trg_w):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img

    def __getitem__(self, index: int | slice | Sequence[int]):

        sample = self.data[index]
        image = self.load_image(sample['image'], self.h, self.w)
        image = IMAGE_TRANSFORMS(image)
        sample['images'] = image

        masked = self.load_image(sample['masked'], self.h, self.w)
        masked = IMAGE_TRANSFORMS(masked)
        sample['mask_imgs'] = masked


        #caption = sample['caption']
        caption = sample['class_caption']
        input_ids, caption_attention_mask = self.get_input_ids(caption, self.tokenizers[0])
        sample['input_ids'] = input_ids

        class_caption = sample['class_caption']
        trg_indexs = self.generate_text_embedding(caption, class_caption, self.tokenizers[0])
        trg_indexs = [1.1,2.1]
        sample['trg_indexs_list'] = torch.Tensor(trg_indexs)
        return sample

