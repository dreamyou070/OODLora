from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
#from pytorch_lightning import Trainer
from torchvision.transforms import ToPILImage
from src.anomalib.config import get_configurable_parameters
from src.anomalib.data import get_datamodule
from src.anomalib.data.utils import read_image
from src.anomalib.deploy import OpenVINOInferencer
from src.anomalib.models import get_model
from src.anomalib.pre_processing.transforms import Denormalize
from src.anomalib.utils.callbacks import LoadModelCallback, get_callbacks

def main() :

    print(f'\n step 1. set directory')
    current_directory = Path.cwd()
    if current_directory.name == "000_getting_started":
        root_directory = current_directory.parent.parent
    elif current_directory.name == "anomalib":
        root_directory = current_directory
    os.chdir(root_directory)

    print(f'\n step 2. set directory')
    #MODEL = "padim"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
    #CONFIG_PATH = root_directory / f"src/anomalib/models/{MODEL}/config.yaml"
    #with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
    #    print(file.read())


if __name__ == '__main__':
    main()