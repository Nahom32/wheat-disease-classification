# Wheat Disease Classification

Multi-label wheat disease classification using deep neural networks on the WFD-2020 dataset (RGB images). Supports multiple architectures via an extensible, config-driven pipeline.

## Dataset: WFD-2020

The dataset contains RGB images of wheat plants with **7 multi-label classes**: `leaf_rust`, `stem_rust`, `yellow_rust`, `powdery_mildew`, `septoria`, `healthy`, `seedlings`. Multiple diseases can co-exist in a single image, making it a multi-label classification problem (BCEWithLogitsLoss + sigmoid, **not** softmax).

## Project Structure

```
├── configs/           # Experiment configuration dataclasses
├── data/              # Dataset class & augmentation presets
├── models/            # Model builder (timm factory)
├── training/          # Trainer, training utilities
├── evaluation/        # Metrics & plotting (loss curves, F1, PR, CM)
├── experiments/       # Experiment registry
├── pipeline.py        # CLI entry point
└── run_pipeline.sh    # Shell launcher
```

See `architecture.md` for full details on design decisions and how to extend the project.

## Setup

```bash
pip install -r requirements.txt
```

Update paths in the config files (`configs/*.py`) to point to your local copies of:
- `wfd_dataset/` (images)
- `data_train.csv`, `data_valid.csv`, `data_test.csv`

## Usage

```bash
# Run all experiments
python pipeline.py --all

# Run specific experiments
python pipeline.py --experiments efficientnet_b4 convnext_small

# Override random seed
python pipeline.py --experiments efficientnet_b4 --seed 123

# Via shell script
./run_pipeline.sh efficientnet_b4
```

## Experiments

| Experiment | Model | Image Size | Augmentations | Source Notebook |
|---|---|---|---|---|
| `efficientnet_b4` | EfficientNet-B4 | 380 | HorizontalFlip, RandomBrightnessContrast | `notebooks/Classification_Models_for_wheat_disease.ipynb` |
| `convnext_small` | ConvNeXt-Small (fb_in22k_ft_in1k) | 224 | RandomResizedCrop, Transpose, H/VFlip, ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast | `notebooks/ConvNeXt_Fine_Tuning_Pipeline.ipynb` |

## Outputs

All artifacts are written under `outputs/`:

```
outputs/
├── checkpoints/           # Best model weights (*.pth)
│   ├── efficientnet_b4_img380_bs16_lr0.0001/
│   └── convnext_small_img224_bs8_lr2e-05/
├── logs/                  # Training history & experiment summaries (JSON)
└── figures/               # Evaluation plots
    ├── efficientnet_b4_.../
    │   ├── loss_curves.png
    │   ├── f1_micro_curve.png
    │   ├── f1_per_class.png
    │   ├── pr_curves.png
    │   └── confusion_matrices.png
    └── convnext_small_.../
```

## Adding a New Experiment

1. Create `configs/my_model.py` with a config dataclass inheriting `BaseConfig`
2. Register it in `experiments/registry.py` via `register('my_model', MyModelConfig, Trainer)`
3. Run: `python pipeline.py --experiments my_model`

For custom training logic, subclass `Trainer` and register your custom trainer instead.

## Citation

If you use this code, please cite the original WFD-2020 dataset authors.
