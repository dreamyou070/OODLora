from __future__ import annotations
import os
from pathlib import Path
import argparse
from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
#from pytorch_lightning import Trainer
from torchvision.transforms import ToPILImage
from src.config import get_configurable_parameters
from src.data import get_datamodule
from src.data.utils import read_image
from src.deploy import OpenVINOInferencer
from src.models import get_model
from src.pre_processing.transforms import Denormalize
from src.utils.callbacks import LoadModelCallback, get_callbacks

def main(args) :

    print(f'\n step 1. set directory')
    current_directory = Path.cwd()

    print(f'\n step 2. get model (with configuration)')
    MODEL = args.model
    CONFIG_PATH = f"./src/models/{MODEL}/config.yaml"
    with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
        print(file.read())


    print(f'\n step 3. get dataset')
    config = get_configurable_parameters(config_path=CONFIG_PATH)
    print(f' - dataconfig : {config}')
    datamodule = get_datamodule(config = config)
    print(f' (3.1) download dataset')
    print(f' (3.1.1) check where to save data')
    datamodule.root = r'/data7/sooyeon/MyData/anomaly_detection'
    datamodule.prepare_data()
    datamodule.setup()  # Create train/val/test/prediction sets.
    #i, data = next(enumerate(datamodule.val_dataloader()))
    #print(data.keys())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = 'padim',
                        choices = ['padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'])
    args = parser.parse_args()
    main(args)