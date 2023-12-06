import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from typing import List
import re

def load_image(path):
    """
    Load image from path
    :param path: path to image
    :return: image
    """
    return cv2.imread(path)

def grid_path(path_list:List[str], image_per_row:int=5):
    """
    Grid images in a list of paths.
    """
    assert len(path_list) > 0, 'Empty path list.'
    assert image_per_row > 0, 'Invalid number of images per row.'
    images = [load_image(path) for path in path_list]
    grid_image = grid_image_from_images(images, image_per_row)
    return grid_image

def grid_image_from_images(images:List[np.ndarray], image_per_row:int=5):
    """
    Grid images in a list of images.
    """
    assert len(images) > 0, 'Empty image list.'
    assert image_per_row > 0, 'Invalid number of images per row.'
    image_per_row = min(image_per_row, len(images))
    image_height = min([image.shape[0] for image in images])
    image_width = min([image.shape[1] for image in images])
    resize_images = [cv2.resize(image, (image_width, image_height)) for image in images]
    image_per_col = int(np.ceil(len(images) / image_per_row))
    grid_image = np.zeros((image_per_col * image_height, image_per_row * image_width, 3), dtype=np.uint8)
    for i, image in enumerate(resize_images):
        row = i // image_per_row
        col = i % image_per_row
        grid_image[row * image_height:(row + 1) * image_height, col * image_width:(col + 1) * image_width, :] = image
    return grid_image

def glob_path(path:str):
    """
    Glob path.
    """
    # e000001_01_41 -> e<epoch>_<setting>_<seed>
    # group by <setting>_<seed>
    # sort by <epoch>
    all_paths = glob.glob(path + os.path.sep + '**.png')
    # get groups
    groups = {}
    for path in all_paths:
        base_name = os.path.basename(path)
        pure_name = os.path.splitext(base_name)[0]
        # regex for e000000_00_00
        pattern = r'e(\d+)_(\d+)_(\d+)'
        match = re.search(pattern, pure_name)
        if match:
            epoch = int(match.group(1))
            setting = int(match.group(2))
            seed = int(match.group(3))
            key_to_use = f'{setting}_{seed}'
            if key_to_use not in groups:
                groups[key_to_use] = []
            groups[key_to_use].append((epoch, path))
    # sort groups
    for keys in groups:
        groups[keys] = sorted(groups[keys], key=lambda x: x[0])
    # flatten groups
    return list(groups.values())
            

def process_path(path:str) -> List[str]:
    """
    Process path and save as separate images.
    """
    result_paths = []
    for idx, image_paths in enumerate(glob_path(path)):
        image_paths = [path for epoch, path in image_paths]
        result = grid_path(image_paths)
        cv2.imwrite(f'{path}{os.path.sep}{idx}.png', result)
        result_paths.append(f'{path}{os.path.sep}{idx}.png')
    return result_paths
