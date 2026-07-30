#!/usr/bin/env python3
"""Main pipeline: run one or more experiments, compare results.

Usage:
    python pipeline.py --experiments efficientnet_b4 convnext_small
    python pipeline.py --experiments efficientnet_b4 --seed 123
    python pipeline.py --all
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.registry import get_experiment, list_experiments
from data.dataset import WFD2020Dataset, get_train_transforms, get_eval_transforms
from models.builder import build_model
from evaluation.plotting import generate_report


def run_experiment(name, seed=None):
    config_cls, trainer_cls = get_experiment(name)
    cfg = config_cls()
    if seed is not None:
        cfg.seed = seed

    exp_name = cfg.experiment_name()
    print(f"\n{'='*60}")
    print(f"Running experiment: {name} ({exp_name})")
    print(f"{'='*60}")

    # Data
    train_df = pd.read_csv(cfg.train_csv)
    valid_df = pd.read_csv(cfg.valid_csv)
    test_df = pd.read_csv(cfg.test_csv)

    train_dataset = WFD2020Dataset(train_df, cfg.image_dir, transforms=get_train_transforms(cfg))
    valid_dataset = WFD2020Dataset(valid_df, cfg.image_dir, transforms=get_eval_transforms(cfg))
    test_dataset = WFD2020Dataset(test_df, cfg.image_dir, transforms=get_eval_transforms(cfg))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}")

    # Model
    model = build_model(cfg)

    # Trainer
    trainer = trainer_cls(cfg, model, train_loader, val_loader, test_loader)
    history = trainer.fit()

    # Test evaluation with full metrics + plots
    test_loss, test_f1_micro, test_f1_macro, test_preds, test_labels = trainer.validate(
        loader=test_loader
    )
    # Re-run test with sigmoid probs for PR curves
    trainer.model.eval()
    test_probs = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(cfg.device)
            outputs = trainer.model(images)
            test_probs.append(torch.sigmoid(outputs).cpu().numpy())
    test_probs = np.concatenate(test_probs)

    f1_dict = generate_report(cfg, history, test_labels, test_preds, test_probs)

    trainer.save_history()

    summary = {
        'experiment': name,
        'config': cfg.to_dict(),
        'test_f1_micro': float(test_f1_micro),
        'test_f1_macro': float(test_f1_macro),
        'per_class_f1': f1_dict,
    }
    summary_path = os.path.join(cfg.output_dir, 'logs', f'{exp_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment {name} complete. Test F1 Micro: {test_f1_micro:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='WFD-2020 Wheat Disease Classification Pipeline')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiments', nargs='+', help='Experiment names to run')
    group.add_argument('--all', action='store_true', help='Run all registered experiments')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    args = parser.parse_args()

    if args.all:
        exp_names = list_experiments()
    else:
        exp_names = args.experiments

    results = []
    for name in exp_names:
        summary = run_experiment(name, seed=args.seed)
        results.append(summary)

    # Comparison table
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'F1 Micro':<12} {'F1 Macro':<12}")
    print('-' * 49)
    for r in results:
        print(f"{r['experiment']:<25} {r['test_f1_micro']:<12.4f} {r['test_f1_macro']:<12.4f}")


if __name__ == '__main__':
    main()
