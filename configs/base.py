from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import torch


@dataclass
class BaseConfig:
    # Paths
    image_dir: str = '/content/drive/MyDrive/'
    train_csv: str = '/content/drive/MyDrive/data_train.csv'
    valid_csv: str = '/content/drive/MyDrive/data_valid.csv'
    test_csv: str = '/content/drive/MyDrive/data_test.csv'
    output_dir: str = 'outputs'

    # Model
    model_name: str = 'efficientnet_b4'
    num_classes: int = 7
    pretrained: bool = True

    # Training
    image_size: int = 380
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_workers: int = 2

    # Augmentations
    train_aug: str = 'default_train'    # key for augmentation preset
    eval_aug: str = 'default_eval'      # key for augmentation preset

    # Mixed precision
    use_amp: bool = False

    # Scheduler
    scheduler: str = 'none'             # 'cosine' or 'none'
    warmup_epochs: int = 0

    # Loss
    loss: str = 'BCEWithLogitsLoss'

    # Misc
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_best: bool = True
    monitor_metric: str = 'f1_micro'
    monitor_mode: str = 'max'

    # Labels — keep consistent order across all experiments
    labels: List[str] = field(default_factory=lambda: [
        'leaf_rust', 'stem_rust', 'yellow_rust', 'powdery_mildew',
        'septoria', 'healthy', 'seedlings'
    ])

    def to_dict(self) -> dict:
        return asdict(self)

    def experiment_name(self) -> str:
        return f"{self.model_name}_img{self.image_size}_bs{self.batch_size}_lr{self.lr}"
