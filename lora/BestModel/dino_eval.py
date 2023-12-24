import os
import argparse
import torch

def main() :

    model_dir = r'../'

    vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

if __name__ == '__main__':
    main()