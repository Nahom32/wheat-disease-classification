"""Experiment registry — maps experiment names to (config, trainer) pairs.

Usage:
    from experiments.registry import get_experiment
    config_cls, trainer_cls = get_experiment('efficientnet_b4')
"""

from configs.efficientnet_b4 import EfficientNetB4Config
from configs.convnext_small import ConvNeXtSmallConfig
from configs.convnext_medium import ConvNeXtMediumConfig
from training.trainer import Trainer

_registry = {}


def register(name, config_cls, trainer_cls=Trainer):
    _registry[name] = (config_cls, trainer_cls)


def get_experiment(name):
    if name not in _registry:
        raise KeyError(
            f"Unknown experiment '{name}'. Available: {list(_registry.keys())}"
        )
    return _registry[name]


def list_experiments():
    return list(_registry.keys())


register('efficientnet_b4', EfficientNetB4Config, Trainer)
register('convnext_small', ConvNeXtSmallConfig, Trainer)
register('convnext_medium', ConvNeXtMediumConfig, Trainer)
