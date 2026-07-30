import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class WFD2020Dataset(Dataset):
    """Shared dataset for WFD-2020 multi-label wheat disease classification.

    Expects CSV files with columns: img, healthy, leaf_rust, powdery_mildew,
    seedlings, septoria, stem_rust, yellow_rust.
    """

    def __init__(self, df: pd.DataFrame, image_dir: str, transforms=None):
        self.image_ids = df['img'].values
        self.labels = df.drop(columns=['img']).values.astype(float)
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, label


def get_train_transforms(cfg):
    """Select augmentation preset based on config key."""
    presets = {
        'default_train': A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        'efficientnet_train': A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        'convnext_train': A.Compose([
            A.RandomResizedCrop(height=cfg.image_size, width=cfg.image_size, p=1.0),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ]),
    }
    return presets.get(cfg.train_aug, presets['default_train'])


def get_eval_transforms(cfg):
    return A.Compose([
        A.Resize(cfg.image_size, cfg.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
