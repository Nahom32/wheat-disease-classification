from dataclasses import dataclass
from configs.base import BaseConfig


@dataclass
class ConvNeXtSmallConfig(BaseConfig):
    """Configuration replicating ConvNeXt_Fine_Tuning_Pipeline.ipynb"""
    model_name: str = 'convnext_small.fb_in22k_ft_in1k'
    image_size: int = 224
    batch_size: int = 8
    epochs: int = 20
    lr: float = 2e-5
    weight_decay: float = 0.05
    num_workers: int = 8
    scheduler: str = 'cosine'
    use_amp: bool = True
    train_aug: str = 'convnext_train'
    eval_aug: str = 'default_eval'
