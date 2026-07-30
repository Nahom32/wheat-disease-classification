# Wheat Disease Classification

Multi-label wheat disease classification using deep neural networks on the WFD-2020 dataset (RGB images). Supports multiple architectures via an extensible, config-driven pipeline with built-in benchmarking.

## Dataset: WFD-2020

The dataset contains RGB images of wheat plants with **7 multi-label classes**: `leaf_rust`, `stem_rust`, `yellow_rust`, `powdery_mildew`, `septoria`, `healthy`, `seedlings`. Multiple diseases can co-exist in a single image, making it a multi-label classification problem (BCEWithLogitsLoss + sigmoid, **not** softmax).

## Project Structure

```
├── configs/           # Experiment configuration dataclasses
├── data/              # Dataset, augmentations, Google Drive downloader
├── models/            # Model builder (timm factory)
├── training/          # Trainer, reproducibility utilities
├── evaluation/        # Metrics, plotting, benchmarking (CSV + JSON)
├── experiments/       # Experiment registry
├── outputs/           # All generated artifacts (gitignored)
│   ├── checkpoints/   # Best model weights
│   ├── logs/          # Training history & experiment summaries
│   ├── figures/       # Per-experiment & cross-experiment plots
│   └── benchmark/     # Comparison CSV and JSON across experiments
├── pipeline.py        # CLI entry point
└── run_pipeline.sh    # Shell launcher
```

See `architecture.md` for full details on design decisions and how to extend the project.

## Setup

```bash
pip install -r requirements.txt
```

### Option A: Download from Google Drive (recommended)

```bash
python pipeline.py --download
```

This downloads the image folder and CSVs to `data/wfd/`. Requires `gdown` (included in `requirements.txt`).

### Option B: Manual paths

Update paths in the config files (`configs/*.py`) to point to your local copies of:
- `wfd_dataset/` (images)
- `data_train.csv`, `data_valid.csv`, `data_test.csv`

## Usage

```bash
# Download data + run all experiments
python pipeline.py --download --all

# Run specific experiments
python pipeline.py --experiments efficientnet_b4 convnext_small convnext_medium

# Override random seed
python pipeline.py --experiments efficientnet_b4 --seed 123

# Download only (no training)
python pipeline.py --download

# Via shell script (downloads + runs all by default)
./run_pipeline.sh
./run_pipeline.sh --no-download efficientnet_b4
```

## Experiments

| Experiment | Model | Params | Image Size | Logits | Augmentations | Source Notebook |
|---|---|---|---|---|---|---|
| `efficientnet_b4` | EfficientNet-B4 | ~19M | 380 | BCE + sigmoid | HorizontalFlip, RandomBrightnessContrast | `notebooks/Classification_Models_for_wheat_disease.ipynb` |
| `convnext_small` | ConvNeXt-Small (fb_in22k_ft_in1k) | ~50M | 224 | BCE + sigmoid | RandomResizedCrop, Transpose, H/VFlip, ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast | `notebooks/ConvNeXt_Fine_Tuning_Pipeline.ipynb` |
| `convnext_medium` | ConvNeXt-Base (fb_in22k_ft_in1k) | ~89M | 224 | BCE + sigmoid | Same as ConvNeXt Small | — |

## Benchmarking

After all experiments complete, the pipeline automatically generates:

**`outputs/benchmark/`**
- `experiment_comparison.csv` — one row per experiment with F1 micro, macro, and per-class scores
- `experiment_comparison.json` — same data in JSON format

**`outputs/figures/comparison/`**
- `f1_comparison.png` — grouped bar chart comparing F1 micro/macro across experiments
- `per_class_comparison.png` — grouped bar chart comparing per-class F1 across experiments

## Outputs

```
outputs/
├── benchmark/
│   ├── experiment_comparison.csv
│   └── experiment_comparison.json
├── checkpoints/
│   ├── efficientnet_b4_img380_bs16_lr0.0001/
│   ├── convnext_small_img224_bs8_lr2e-05/
│   └── convnext_medium_img224_bs8_lr2e-05/
├── logs/
│   └── *.json              # History + summary per experiment
└── figures/
    ├── comparison/
    │   ├── f1_comparison.png
    │   └── per_class_comparison.png
    ├── efficientnet_b4_.../
    │   ├── loss_curves.png
    │   ├── f1_micro_curve.png
    │   ├── f1_per_class.png
    │   ├── pr_curves.png
    │   └── confusion_matrices.png
    └── convnext_*_.../
```

## Adding a New Experiment

1. Create `configs/my_model.py` with a config dataclass inheriting `BaseConfig`
2. Register it in `experiments/registry.py` via `register('my_model', MyModelConfig, Trainer)`
3. Run: `python pipeline.py --experiments my_model`

For custom training logic, subclass `Trainer` and register your custom trainer instead.

## Citation

If you use this code, please cite the original WFD-2020 dataset authors.
