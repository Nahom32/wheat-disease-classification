# Architecture Overview

## Project Structure

```
wheat-disease-classification/
├── configs/                  # Experiment configuration
│   ├── base.py               # Base configuration class
│   ├── efficientnet_b4.py    # EfficientNet-B4 config (from notebook 1)
│   └── convnext_small.py     # ConvNeXt-Small config (from notebook 2)
├── data/                     # Data loading
│   ├── __init__.py
│   └── dataset.py            # WFD2020Dataset — shared across experiments
├── models/                   # Model definitions
│   ├── __init__.py
│   └── builder.py            # Model builder factory
├── training/                 # Training loop
│   ├── __init__.py
│   ├── trainer.py            # Base Trainer class
│   └── utils.py              # Helper functions (mixup, AWP, etc.)
├── evaluation/               # Evaluation & visualization
│   ├── __init__.py
│   ├── metrics.py            # F1, precision-recall, confusion matrix
│   └── plotting.py           # Loss curves, per-class F1, PR, CM plots
├── experiments/              # Runnable experiment scripts
│   ├── __init__.py
│   └── registry.py           # Experiment registry
├── outputs/                  # All outputs (gitignored)
│   ├── checkpoints/          # Model weights
│   ├── logs/                 # Training logs
│   └── figures/              # Evaluation figures
├── notebooks/                # Original notebooks (kept for reference)
│   ├── Classification_Models_for_wheat_disease.ipynb
│   └── ConvNeXt_Fine_Tuning_Pipeline.ipynb
├── pipeline.py               # Main pipeline runner
├── run_pipeline.sh           # Shell script to execute the pipeline
├── requirements.txt          # Python dependencies
├── architecture.md           # This file
└── README.md                 # Project README
```

## Architectural Decisions

### 1. Config-Driven Experiments
Every experiment is defined by a config class (inheriting from `BaseConfig`) that specifies model name, image size, batch size, learning rate, augmentations, etc. This makes it trivial to add new experiments — just create a new config and register it.

### 2. Shared Dataset
The `WFD2020Dataset` class in `data/dataset.py` is the single source of truth for loading WFD-2020 images and multi-labels. Both experiments use it. The column names and label order are centralised here.

### 3. Model Builder (Factory)
`models/builder.py` provides a `build_model()` function that takes a config and returns a model. New architectures are added by extending the `create_model()` switch, typically just a `timm.create_model()` call with a custom head swap.

### 4. Extensible Trainer
`training/trainer.py` defines a `Trainer` base class with the standard train/validate loop. It calls configurable hooks:
- `on_epoch_start` / `on_epoch_end`
- `on_batch_start` / `on_batch_end`
- `get_optimizer`, `get_scheduler`, `get_criterion`

Subclass the trainer to override any hook without changing the loop structure.

### 5. Decoupled Evaluation
All metric computation and plotting lives in `evaluation/`. The trainer calls `evaluate()` after each epoch (returning scalars to track), and the post-training `generate_report()` produces all figures and a summary table. This separation lets researchers swap metrics or visualisations without touching training code.

### 6. Experiment Registry
`experiments/registry.py` maps experiment names to `(config_class, trainer_class)` tuples. The pipeline runner selects which experiments to run by name. Adding a new experiment means:
1. Create a config in `configs/`
2. (Optionally) create a custom trainer in `experiments/`
3. Register it in the registry

### 7. Pipeline Orchestrator
`pipeline.py` is the single entry point. It:
- Parses CLI args (`--experiments`, `--seed`, `--data_root`)
- Loads configs
- Iterates over requested experiments
- For each: trains the model, evaluates, generates figures, logs results
- Outputs a comparison table across all experiments

### 8. Versioning & Reproducibility
- A fixed random seed is set at the start of each experiment.
- Configs are serialised to JSON in the output log directory.
- All outputs (checkpoints, logs, figures) are organised under `outputs/{experiment_name}/`.

## How to Add a New Experiment

1. Create `configs/my_model.py`:
   ```python
   from configs.base import BaseConfig

   @dataclass
   class MyModelConfig(BaseConfig):
       model_name: str = 'tf_efficientnetv2_s'
       image_size: int = 384
       batch_size: int = 16
       lr: float = 1e-4
   ```

2. Register in `experiments/registry.py`:
   ```python
   register('my_model', MyModelConfig, Trainer)
   ```

3. Run:
   ```bash
   python pipeline.py --experiments my_model
   ```
