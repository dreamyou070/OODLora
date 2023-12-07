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
from pytorch_lightning import Trainer

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
    # config =  {'dataset': {'name': 'mvtec', 'format': 'mvtec', 'path': './datasets/MVTec', 'category': 'bottle',
    # 'task': 'segmentation', 'train_batch_size': 32, 'eval_batch_size': 32, 'num_workers': 8,
    # 'image_size': [256, 256], 'center_crop': None, 'normalization': 'imagenet',
    # 'transform_config': {'train': None, 'eval': None},
    # 'test_split_mode': 'from_dir', 'test_split_ratio': 0.2,
    # 'val_split_mode': 'same_as_test', 'val_split_ratio': 0.5, 'tiling': {'apply': False, 'tile_size': None,
    # 'stride': None, 'remove_border_count': 0, 'use_random_tiling': False, 'random_tile_count': 16}},
    # 'model': {'name': 'padim', 'backbone': 'resnet18', 'pre_trained': True, 'layers': ['layer1', 'layer2', 'layer3'],
    # 'normalization_method': 'min_max', 'input_size': [256, 256]}, 'metrics': {'image': ['F1Score', 'AUROC'], 'pixel': ['F1Score', 'AUROC'],
    # 'threshold': {'method': 'adaptive', 'manual_image': None, 'manual_pixel': None}},
    # 'visualization': {'show_images': False, 'save_images': True, 'log_images': True, 'image_save_path': None, 'mode': 'full'},
    # 'project': {'seed': 42, 'path': 'results/padim/mvtec/bottle/run', 'unique_dir': False},
    # 'logging': {'logger': [], 'log_graph': False},
    # 'optimization': {'export_mode': None}, ---------------------> change to ['openvino']
    # 'trainer': {'enable_checkpointing': True,
    # 'default_root_dir': 'results/padim/mvtec/bottle/run',
    # 'gradient_clip_val': 0, 'gradient_clip_algorithm': 'norm', 'num_nodes': 1,
    # 'devices': 1, 'enable_progress_bar': True, 'overfit_batches': 0.0, 'track_grad_norm': -1,
    # 'check_val_every_n_epoch': 1, 'fast_dev_run': False, 'accumulate_grad_batches': 1, 'max_epochs': 1, 'min_epochs': None, 'max_steps': -1,
    # 'min_steps': None, 'max_time': None, 'limit_train_batches': 1.0,
    # 'limit_val_batches': 1.0, 'limit_test_batches': 1.0, 'limit_predict_batches': 1.0, 'val_check_interval': 1.0, 'log_every_n_steps': 50,
    # 'accelerator': 'auto', 'strategy': None, 'sync_batchnorm': False, 'precision': 32, 'enable_model_summary': True, 'num_sanity_val_steps': 0,
    # 'profiler': None, 'benchmark': False, 'deterministic': False, 'reload_dataloaders_every_n_epochs': 0,
    # 'auto_lr_find': False, 'replace_sampler_ddp': True, 'detect_anomaly': False, 'auto_scale_batch_size': False, 'plugins': None,
    # 'move_metrics_to_cpu': False, 'multiple_trainloader_mode': 'max_size_cycle'}}
    print(f' (3.1) download dataset (check where to save data')
    config.dataset.path = r'/data7/sooyeon/MyData/anomaly_detection'
    datamodule = get_datamodule(config=config)
    datamodule.prepare_data()
    datamodule.setup()  # Create train/val/test/prediction sets.

    print(f' (3.2) validation check')
    i, data = next(enumerate(datamodule.val_dataloader()))
    print(data.keys())
    print(data["image"].shape, data["mask"].shape)

    print(f'\n step 4. prepare model and callbacks')
    # Set the export-mode to OpenVINO to create the OpenVINO IR model.
    config.optimization.export_mode = "openvino"
    model = get_model(config)
    callbacks = get_callbacks(config)

    print(f'\n step 5. start training')
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)

    print(f'\n step 6. validation')
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)
    test_results = trainer.test(model=model,
                                datamodule=datamodule)
    print(f'\n step 7. Load the OpenVINO Model')
    output_path = Path(config["project"]["path"])
    print(output_path)

    print(f'\n step 8. inference')
    inferencer = OpenVINOInferencer(
        path=openvino_model_path,  # Path to the OpenVINO IR model.
        metadata=metadata,  # Path to the metadata file.
        device="CPU",  # We would like to run it on an Intel CPU.
    )



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = 'padim',
                        choices = ['padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'])
    args = parser.parse_args()
    main(args)