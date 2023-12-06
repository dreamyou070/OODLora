import os, torch
from safetensors.torch import load_file, safe_open

pretrained_lora_dir = 'PlatformGameV0_2.safetensors'
weights_sd = load_file(pretrained_lora_dir)
for layer in weights_sd.keys():

    if "alpha" in layer :
        alpha_value = weights_sd[layer]
    elif "lora_down" in layer :
        layer_name = layer.split('.lora')[0]
        down_weight = weights_sd[layer].type(torch.float32)
        up_layer_key = f'{layer_name}.lora_up.weight'
        up_weight = weights_sd[up_layer_key].type(torch.float32)
        if len(down_weight.size()) == 2:
            weight = (up_weight @ down_weight)
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else :
            weight = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
        mean = weight.mean()
        std = weight.std()
        print(f'layer_name : {layer_name} | lora weight mean : {mean} | std : {std}')