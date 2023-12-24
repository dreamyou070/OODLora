import os
import argparse
import torch
from vision_transformer import vit_tiny
def main(args) :

    print(f'step 1. load pretrained model')
    model_dir = args.model_dir
    model_state = torch.load(model_dir, map_location="cpu")
    student_model_state_dict = model_state['student']
    teacher_model_state_dict = model_state['teacher']

    print(f'step 2. model')
    smal_model = vit_tiny()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINO evaluation script')
    parser.add_argument('--model_dir', default=r'../../../../pretrained_models/dino_deitsmall16_pretrain_full_checkpoint.pth',
                        type=str, help='path to the model')
    args = parser.parse_args()
    main(args)