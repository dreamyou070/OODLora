import torch
from typing import Union

def get_state_dict(dir):
    model_state_dict = torch.load(dir, map_location="cpu")
    state_dict = {}
    for k, v in model_state_dict.items():
        k_ = '.'.join(k.split('.')[1:])
        state_dict[k_] = v
    return state_dict


def get_crossattn_map(args, attention_stores: dict = None,
                      trg_layer: str = 'up_blocks_3_attentions_0_transformer_blocks_0_attn2',
                      res : int = 64,
                      thredhold: float = 0.5):
    """ 'up_blocks_3_attentions_0_transformer_blocks_0_attn2'"""
    attn = attention_stores[trg_layer][0].squeeze()  # head, pix_num
    if args.truncate_length == 3:
        cls_score, trigger_score, pad_score = attn.chunk(3, dim=-1)  # head, pix_num
    else:
        cls_score, trigger_score = attn.chunk(2, dim=-1)  # head, pix_num
    h = trigger_score.shape[0]
    trigger_score = trigger_score.unsqueeze(-1).reshape(h, res,res)
    trigger_score = trigger_score.mean(dim=0)  # res, res, (object = 1)
    object_mask = trigger_score / trigger_score.max()
    object_mask = torch.where(object_mask > thredhold, 1, 0)  # res, res, (object = 1)
    return object_mask

def init_prompt(tokenizer, text_encoder, device, prompt: str,
                negative_prompt: Union[str, None] = None):
    if negative_prompt :
        n_p = negative_prompt
    else :
        n_p = ""
    uncond_input = tokenizer([n_p],
                             padding="max_length", max_length=tokenizer.model_max_length,
                             return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_input = tokenizer([prompt],
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt",)
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