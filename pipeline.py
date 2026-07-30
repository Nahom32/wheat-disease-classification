#!/usr/bin/env python3
"""Main pipeline: download data, run experiments, compare results.

Usage:
    python pipeline.py --download                          # download data only
    python pipeline.py --experiments efficientnet_b4       # run specific
    python pipeline.py --all                               # run all
    python pipeline.py --download --all                    # download + run all
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
from evaluation.plotting import generate_report, plot_comparison_bar, plot_per_class_comparison
from evaluation.benchmark import write_benchmark


def resolve_paths(cfg, data_root=None):
    """Update config paths if a local data root is provided."""
    if data_root is None:
        return cfg
    cfg.image_dir = os.path.join(data_root, "wfd_dataset")
    cfg.train_csv = os.path.join(data_root, "csv", "data_train.csv")
    cfg.valid_csv = os.path.join(data_root, "csv", "data_valid.csv")
    cfg.test_csv = os.path.join(data_root, "csv", "data_test.csv")
    return cfg


def download_data(data_root: str):
    """Download the WFD-2020 dataset if it doesn't already exist."""
    from data.downloader import download_wfd_dataset

    images_dir = os.path.join(data_root, "wfd_dataset")
    csv_dir = os.path.join(data_root, "csv")

    images_exist = os.path.isdir(images_dir) and bool(os.listdir(images_dir))
    csv_exist = (
        os.path.isdir(csv_dir)
        and all(
            os.path.isfile(os.path.join(csv_dir, f))
            for f in ["data_train.csv", "data_valid.csv", "data_test.csv"]
        )
    )

    if images_exist and csv_exist:
        print("Dataset already downloaded, skipping.")
        return

    download_wfd_dataset(
        data_root=data_root,
        download_images=not images_exist,
        download_csv=not csv_exist,
    )


def run_experiment(name, seed=None, data_root=None):
    config_cls, trainer_cls = get_experiment(name)
    cfg = config_cls()
    cfg = resolve_paths(cfg, data_root)
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
    parser.add_argument('--experiments', nargs='+', help='Experiment names to run')
    parser.add_argument('--all', action='store_true', help='Run all registered experiments')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset from Google Drive before running')
    parser.add_argument('--data-root', default=None,
                        help='Local data directory (overrides config paths)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    args = parser.parse_args()

    # Resolve data root
    data_root = args.data_root
    if args.download:
        download_data(data_root or 'data/wfd')
        if data_root is None:
            data_root = 'data/wfd'

    # Determine experiments to run
    if args.all:
        exp_names = list_experiments()
    elif args.experiments:
        exp_names = args.experiments
    else:
        return  # download-only mode

    results = []
    labels = None
    for name in exp_names:
        summary = run_experiment(name, seed=args.seed, data_root=data_root)
        results.append(summary)
        if labels is None and 'config' in summary:
            labels = summary['config'].get('labels', [])

    if len(results) > 0:
        # Persistent benchmark files (CSV + JSON)
        write_benchmark(results)

        # Cross-experiment comparison plots
        figure_dir = os.path.join('outputs', 'figures', 'comparison')
        os.makedirs(figure_dir, exist_ok=True)
        plot_comparison_bar(results, os.path.join(figure_dir, 'f1_comparison.png'))
        if labels:
            plot_per_class_comparison(results, labels,
                                      os.path.join(figure_dir, 'per_class_comparison.png'))
        print(f"Comparison figures saved to {figure_dir}/")

    # Console comparison table
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'F1 Micro':<12} {'F1 Macro':<12}")
    print('-' * 49)
    for r in results:
        print(f"{r['experiment']:<25} {r['test_f1_micro']:<12.4f} {r['test_f1_macro']:<12.4f}")


if __name__ == '__main__':
    main()
