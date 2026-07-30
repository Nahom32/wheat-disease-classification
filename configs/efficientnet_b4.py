from dataclasses import dataclass
from configs.base import BaseConfig


@dataclass
class EfficientNetB4Config(BaseConfig):
    """Configuration replicating Classification_Models_for_wheat_disease.ipynb"""
    model_name: str = 'efficientnet_b4'
    image_size: int = 380
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_workers: int = 2
    scheduler: str = 'none'
    use_amp: bool = False
    train_aug: str = 'efficientnet_train'
    eval_aug: str = 'default_eval'
