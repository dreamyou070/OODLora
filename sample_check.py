import os
import torch
flip_times = torch.arange(0, 1000, 20) # [0,20, ..., 980]
print(flip_times)
for ii, final_time in enumerate(flip_times[1:]):
    inference_times = flip_times[:ii + 2]
    print(inference_times)