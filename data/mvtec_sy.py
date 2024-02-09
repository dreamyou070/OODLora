import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import random
import glob
from data.perlin import rand_perlin_2d_np
from PIL import Image
from torchvision import transforms
from numpy.random import default_rng
import cv2
import skimage.exposure

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample


class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self,
                 root_dir,
                 anomaly_source_path,
                 resize_shape=None,
                 tokenizer=None,
                 caption: str = None,
                 use_perlin: bool = False,
                 num_repeat: int = 1,
                 anomal_only_on_object : bool = True):

        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.anomaly_source_paths = []
        for ext in ["png", "jpg"]:
            self.anomaly_source_paths.extend(sorted(glob.glob(anomaly_source_path + f"/*/*/*.{ext}")))

        self.caption = caption
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]),])
        self.use_perlin = use_perlin
        self.num_repeat = 1

        image_paths = sorted(glob.glob(root_dir + "/*.png"))
        self.image_paths = [image_path for image_path in image_paths for i in range(num_repeat)]
        random.shuffle(self.image_paths)
        self.anomal_only_on_object = anomal_only_on_object

    def __len__(self):
        return max(len(self.image_paths), len(self.anomaly_source_paths))
    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True,return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask
    def make_random_mask(self, height, width) -> np.ndarray :

        if self.use_perlin:
            perlin_scale = 6
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            noise = rand_perlin_2d_np((height, width), (perlin_scalex, perlin_scaley))
        else:
            random_generator = default_rng(seed=42)
            noise = random_generator.integers(0, 255, (height, width), np.uint8, True)
        blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
        stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
        thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_pil = Image.fromarray(mask).convert('L')
        mask_np = np.array(mask_pil) / 255  # height, width, [0,1]
        return mask_np, mask_pil

    def load_image(self, image_path, trg_h, trg_w):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img

    def __getitem__(self, idx):

        # idx = torch.randint(0, len(self.image_paths), (1,)).item()
        # anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        img_idx = idx % len(self.image_paths)

        # [1] base
        img_path = self.image_paths[img_idx]
        # img_path = self.image_paths[idx]
        parent, name = os.path.split(img_path)
        parent, _ = os.path.split(parent)
        object_mask_dir = os.path.join(parent, f"object_mask/{name}")

        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1])

        object_img = Image.open(object_mask_dir).convert("L").resize((self.resize_shape[0], self.resize_shape[1]), Image.BICUBIC)
        object_mask_np = np.array(object_img, np.uint8) / 255
        object_mask_np = np.where(object_mask_np == 0, 0, 1)

        # [2] object mask
        object_img_latent =Image.open(object_mask_dir).convert("L").resize((64,64), Image.BICUBIC)
        object_mask_latent_np = np.array(object_img_latent, np.uint8) / 255 # [64,64], 0~1
        object_mask_latent_np = np.where(object_mask_latent_np == 0, 0, 1)  #
        object_mask = torch.tensor(object_mask_latent_np) #

        anomal_src = self.load_image(self.anomaly_source_paths[anomaly_source_idx], self.resize_shape[0], self.resize_shape[1])
        dtype = img.dtype

        # [3] augment ( anomaly mask white = anomal position )
        anomal_mask_np, anomal_mask_pil = self.make_random_mask(self.resize_shape[0], self.resize_shape[1]) # [512, 512], [0, 1]
        anomal_mask_np = np.where(anomal_mask_np == 0, 0, 1)  # strict anomal (0, 1
        if self.anomal_only_on_object:
            anomal_mask_np = anomal_mask_np * object_mask_np
        mask = np.repeat(np.expand_dims(anomal_mask_np, axis=2), 3, axis=2).astype(dtype)
        anomal_img = (1-mask) * img + mask * anomal_src

        # [4] masked image
        masked_img = (1-mask) * img

        # [3] final
        anomal_mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((64,64)).convert('L')
        anomal_mask_np = np.array(anomal_mask_pil) / 255
        anomal_mask_torch = torch.tensor(anomal_mask_np)
        anomal_mask = torch.where(anomal_mask_torch > 0.5, 1, 0)  # strict anomal

        # [4] caption
        input_ids, attention_mask = self.get_input_ids(self.caption) # input_ids = [77]

        # [5] return
        sample = {'image': self.transform(img),
                  "object_mask": object_mask.unsqueeze(0),  # [1, 64, 64]
                  'augmented_image': self.transform(anomal_img),
                  "anomaly_mask": anomal_mask.unsqueeze(0),  # [1, 64, 64]
                  'masked_image': self.transform(masked_img),
                  'masked_image_mask': anomal_mask.unsqueeze(0),
                  'idx': idx,
                  'input_ids': input_ids.squeeze(0),
                  'caption': self.caption,}
        return sample
