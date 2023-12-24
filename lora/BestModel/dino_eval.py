import os
import argparse
import torch

def main(args) :

    model_dir = args.model_dir
    model_state = torch.load(model_dir, map_location="cpu")
    print(model_state.keys())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINO evaluation script')
    parser.add_argument('--model_dir', default=r'../../../../pretrained_models/dino_deitsmall16_pretrain_full_checkpoint.pth',
                        type=str, help='path to the model')
    args = parser.parse_args()
    main(args)