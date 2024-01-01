import torch
import torch.nn as nn
from torchvision import transforms
from copy import deepcopy

SD14_TO_SD21_RATIO = 1.5

# get token index in text
def get_word_idx(text: str, tgt_word, tokenizer):

    tgt_word = tgt_word.lower()

    # ignore the first and last token
    encoded_text = tokenizer.encode(text)[1:-1]
    encoded_tgt_word = tokenizer.encode(tgt_word)[1:-1]

    # find the idx of target word in text
    first_token_idx = -1
    for i in range(len(encoded_text)):
        if encoded_text[i] == encoded_tgt_word[0]:

            if len(encoded_text) > 0:
                # check the following 
                following_match = True
                for j in range(1, len(encoded_tgt_word)):
                    if encoded_text[i + j] != encoded_tgt_word[j]:
                        following_match = False
                if not following_match:
                    continue
            # for a single encoded idx, just take it
            first_token_idx = i

            break

    assert first_token_idx != -1, "word not in text"

    # add 1 for sot token
    tgt_word_tokens_idx_ls = [i + 1 + first_token_idx for i in range(len(encoded_tgt_word))]

    # sanity check
    encoded_text = tokenizer.encode(text)

    decoded_token_ls = []

    for word_idx in tgt_word_tokens_idx_ls:
        text_decode = tokenizer.decode([encoded_text[word_idx]]).strip("#")
        decoded_token_ls.append(text_decode)

    decoded_tgt_word = "".join(decoded_token_ls)
    
    tgt_word_ls = tgt_word.split(" ")
    striped_tgt_word = "".join(tgt_word_ls).strip("#")

    assert decoded_tgt_word == striped_tgt_word, "decode_text != striped_tar_wd"

    return tgt_word_tokens_idx_ls

# get attn loss by resolution
def get_grounding_loss_by_layer(_gt_seg_list,
                                word_token_idx_ls,
                                res,                # 64,32,16,8
                                input_attn_map_ls,
                                is_training_sd21):
    if is_training_sd21:
        # training with sd21, using resolution 768 = 512 * 1.5 (text dimension change)
        res = int(SD14_TO_SD21_RATIO * res)


    gt_seg_list = deepcopy(_gt_seg_list)

    # reszie gt seg map to the same size with attn map
    resize_transform = transforms.Resize((res, res))
    noun_num = len(gt_seg_list)
    for i in range(len(gt_seg_list)):
        gt_seg_list[i] = resize_transform(gt_seg_list[i])
        gt_seg_list[i] = gt_seg_list[i].squeeze(0) # 1, 1, res, res => 1, 1, res(8,16,32,64), res(8,16,32,64)
        # add binary
        binary = (gt_seg_list[i] > 0.0).float() # 1, res, res
        gt_seg_list[i] = (gt_seg_list[i] > 0.0).float()

    ################### token loss start ###################
    # Following code is adapted from
    # https://github.com/silent-chen/layout-guidance/blob/08b687470f911c7f57937012bdf55194836d693e/utils.py#L27
    token_loss = 0.0
    for attn_map in input_attn_map_ls:
        # len is 3 or 1
        b, H, W, j = attn_map.shape
        for i in range(len(word_token_idx_ls)): # [[word1 token_idx1, word1 token_idx2, ...], [word2 token_idx1, word2 token_idx2, ...]]
            obj_loss = 0.0
            single_word_idx_ls = word_token_idx_ls[i] #[token_idx1, token_idx2, ...]
            mask = gt_seg_list[i]
            for obj_position in single_word_idx_ls:
                # ca map obj shape 8 * 16 * 16
                ca_map_obj = attn_map[:, :, :, obj_position].reshape(b, H, W) # 1, 8, 8
                trg_score =  (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)
                all_score =  ca_map_obj.reshape(b, -1).sum(dim=-1)
                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                obj_loss += (1.0 - torch.mean(activation_value)) ** 2
            token_loss += (obj_loss/len(single_word_idx_ls))
    # normalize with len words
    token_loss = token_loss / len(word_token_idx_ls)
    ################## token loss end ##########################

    ################## pixel loss start ######################
    # average cross attention map on different layers
    avg_attn_map_ls = []
    # input_attn_map_list ?
    for i in range(len(input_attn_map_ls)):
        # len is 1 or 3
        org_map = input_attn_map_ls[i]
        print(f'org_map.shape (head, res*res, 77) : {org_map.shape}')
        map = input_attn_map_ls[i].reshape(-1, res, res, input_attn_map_ls[i].shape[-1]).mean(0)

        avg_attn_map_ls.append(map)
        # [head, res,res,c]
    avg_attn_map = torch.stack(avg_attn_map_ls, dim=0) # head, res, res, 1
    print(f'avg_attn_map.shape (1, res,res, 77) : {avg_attn_map.shape}')
    avg_attn_map = avg_attn_map.sum(0) / avg_attn_map.shape[0] # res,res,1
    print(f'avg_attn_map.shape (res, res, 77) : {avg_attn_map.shape}')
    avg_attn_map = avg_attn_map.unsqueeze(0) # 1, rse,res, 77
    bce_loss_func = nn.BCELoss()
    pixel_loss = 0.0
    for i in range(len(word_token_idx_ls)):

        # token idx
        word_cross_attn_ls = []
        for token_idx in word_token_idx_ls[i]:
            # 2
            # 9
            # 5
            word_map = avg_attn_map[..., token_idx] # 1, res, res, 1
            word_cross_attn_ls.append(word_map)
        word_cross_attn_ls = torch.stack(word_cross_attn_ls, dim=0).sum(dim=0) # 1, rse,res,1
        print(f'word_cross_attn_ls.shape: {word_cross_attn_ls.shape}')

        pixel_loss += bce_loss_func(word_cross_attn_ls, gt_seg_list[i])

    # average with len word_token_idx_ls
    pixel_loss = pixel_loss / len(word_token_idx_ls)
    ################## pixel loss end #########################

    return {
        "token_loss" : token_loss,
        "pixel_loss": pixel_loss,
    }

