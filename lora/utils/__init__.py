from __future__ import annotations
from itertools import chain
from functools import lru_cache
from pathlib import Path
import random
import re

from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
# import spacy
import torch
import torch.nn.functional as F

def expand_image(im: torch.Tensor, h = 512, w = 512,
                 absolute: bool = False, threshold: float = None) -> torch.Tensor:
    im = im.unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im.float().detach(), size=(h, w), mode='bicubic')
    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
    if threshold:
        im = (im > threshold).float()
    # im = im.cpu().detach()
    return im.squeeze()


def _convert_heat_map_colors(heat_map : torch.Tensor):
    def get_color(value):
        return np.array(cm.turbo(value / 255)[0:3])

    color_map = np.array([get_color(i) * 255 for i in range(256) ])
    color_map = torch.tensor(color_map, device=heat_map.device)

    heat_map = (heat_map * 255).long()

    return color_map[heat_map]

def image_overlay_heat_map(img,
                           heat_map,
                           word=None, out_file=None, crop=None, alpha=0.5, caption=None, image_scale=1.0):
    # type: (Image.Image | np.ndarray, torch.Tensor, str, Path, int, float, str, float) -> Image.Image
    assert(img is not None)

    if heat_map is not None:
        shape : torch.Size = heat_map.shape
        # heat_map = heat_map.unsqueeze(-1).expand(shape[0], shape[1], 3).clone()
        heat_map = _convert_heat_map_colors(heat_map)
        heat_map = heat_map.to('cpu').detach().numpy().copy().astype(np.uint8)
        heat_map_img = Image.fromarray(heat_map)
        img = Image.blend(img, heat_map_img, alpha)
    else:
        img = img.copy()

    if caption:
        img = _write_on_image(img, caption)

    if image_scale != 1.0:
        x, y = img.size
        size = (int(x * image_scale), int(y * image_scale))
        img = img.resize(size, Image.BICUBIC)
    return img

from functools import lru_cache
from pathlib import Path
import os
import sys
import random
from typing import TypeVar

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
import torch.nn.functional as F


__all__ = ['set_seed', 'compute_token_merge_indices', 'plot_mask_heat_map', 'cached_nlp', 'cache_dir', 'auto_device', 'auto_autocast']


T = TypeVar('T')


def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

    return obj


def auto_autocast(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs['enabled'] = False

    return torch.cuda.amp.autocast(*args, **kwargs)


def plot_mask_heat_map(im: PIL.Image.Image, heat_map: torch.Tensor, threshold: float = 0.4):
    im = torch.from_numpy(np.array(im)).float() / 255
    mask = (heat_map.squeeze() > threshold).float()
    im = im * mask.unsqueeze(-1)
    plt.imshow(im)


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen


def cache_dir() -> Path:
    # *nix
    if os.name == 'posix' and sys.platform != 'darwin':
        xdg = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        return Path(xdg, 'daa')
    elif sys.platform == 'darwin':
        # Mac OS
        return Path(os.path.expanduser('~'), 'Library/Caches/daa')
    else:
        # Windows
        local = os.environ.get('LOCALAPPDATA', None) \
                or os.path.expanduser('~\\AppData\\Local')
        return Path(local, 'daa')


def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    merge_idxs = []
    tokens = tokenizer.tokenize(prompt.lower())
    if word_idx is None:
        word = word.lower()
        search_tokens = tokenizer.tokenize(word)
        start_indices = [x + offset_idx for x in range(len(tokens)) if tokens[x:x + len(search_tokens)] == search_tokens]
        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]
        if not merge_idxs:
            raise ValueError(f'Search word {word} not found in prompt!')
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.


nlp = None


@lru_cache(maxsize=100000)
def cached_nlp(prompt: str, type='en_core_web_md'):
    global nlp

    if nlp is None:
        try:
            nlp = spacy.load(type)
        except OSError:
            import os
            os.system(f'python -m spacy download {type}')
            nlp = spacy.load(type)

    return nlp(prompt)
