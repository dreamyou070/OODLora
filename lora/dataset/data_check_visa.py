import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusers import StableDiffusionInpaintPipeline

from PIL import Image
import argparse


def main(args):

    base_folder_dir = '/home/dreamyou070/MyData/anomaly_detection/VisA/split_csv'
    files = os.listdir(base_folder_dir)
    for file in files:
        file_dir = os.path.join(base_folder_dir, file)
        with open(file_dir, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            line = line.split(',')
            split = line[1]
            if line[0] == 'candle' and split == 'test' and line[2] == 'normal':
                print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='../../../../MyData/anomaly_detection/VisA')
    args = parser.parse_args()
    main(args)