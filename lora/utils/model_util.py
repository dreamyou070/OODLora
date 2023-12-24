import torch

def get_state_dict(dir):
    model_state_dict = torch.load(dir, map_location="cpu")
    state_dict = {}
    for k, v in model_state_dict.items():
        k_ = '.'.join(k.split('.')[1:])
        state_dict[k_] = v
    return state_dict