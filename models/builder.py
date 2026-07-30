import timm
import torch.nn as nn


def build_model(cfg):
    """Build a model from config using the timm model name.

    Supported model names include:
      - 'efficientnet_b4'
      - 'convnext_small.fb_in22k_ft_in1k'
      - any other timm model name
    """
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
    )
    return model
