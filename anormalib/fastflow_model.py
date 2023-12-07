from __future__ import annotations
from pathlib import Path
from functools import partial, update_wrapper
from types import MethodType
from typing import Any

from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from src.data import InferenceDataset, TaskType
from src.data.mvtec import MVTec
from src.models.fastflow.lightning_model import Fastflow
from src.post_processing import (NormalizationMethod,ThresholdMethod,superimpose_anomaly_map,)
from src.pre_processing.transforms import Denormalize
from src.utils.callbacks import (ImageVisualizerCallback,MetricsConfigurationCallback,MetricVisualizerCallback,
                                 PostProcessingConfigurationCallback,)
import argparse
def main(args) :

    print(f'\n step 1. set what kind of task to do')
    task = TaskType.SEGMENTATION

    print(f'\n step 2. datamodule')
    dataset_root = r'/data7/sooyeon/MyData/anomaly_detection/MVTec'
    datamodule = MVTec(root=dataset_root,
                       category="bottle",
                       image_size=256,
                       train_batch_size=32,
                       eval_batch_size=32,
                       num_workers=8,
                       task=task,)
    datamodule.setup()
    i, data = next(enumerate(datamodule.test_dataloader()))
    print(f'Image Shape: {data["image"].shape} Mask Shape: {data["mask"].shape}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = 'padim',
                        choices = ['padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'])
    parser.add_argument('--experiment_dir', type=str,
                        default= r'/data7/sooyeon/Lora/OODLora/result',)
    args = parser.parse_args()
    main(args)