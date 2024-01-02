import torch
import numpy as np
from typing import Union

def get_state_dict(dir):
    model_state_dict = torch.load(dir, map_location="cpu")
    state_dict = {}
    for k, v in model_state_dict.items():
        k_ = '.'.join(k.split('.')[1:])
        state_dict[k_] = v
    return state_dict

def init_prompt(tokenizer, text_encoder, device, prompt: str):
    uncond_input = tokenizer([""],
                             padding="max_length", max_length=tokenizer.model_max_length,
                             return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_input = tokenizer([prompt],
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt",)
    print(f'prompt : {prompt}, text_input : {text_input}')
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    return context


def call_unet(unet, noisy_latents, timesteps,
              text_conds, trg_indexs_list, mask_imgs):
    noise_pred = unet(noisy_latents,
                      timesteps,
                      text_conds,
                      trg_indexs_list=trg_indexs_list,
                      mask_imgs=mask_imgs,).sample
    return noise_pred