import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix


def plot_loss_curves(history: dict, save_path: str):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_f1_curve(val_f1_scores: list, title: str, save_path: str):
    epochs = range(1, len(val_f1_scores) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_f1_scores, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_f1(f1_dict: dict, save_path: str):
    plt.figure(figsize=(12, 6))
    classes = list(f1_dict.keys())
    scores = list(f1_dict.values())
    plt.bar(classes, scores)
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Per Class (Test Set)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, labels: list, save_path: str):
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Per Class (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, labels: list, save_path: str):
    n = len(labels)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, label in enumerate(labels):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        axes[i].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[i].set_title(f'CM: {label}')
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
        for x in range(2):
            for y in range(2):
                axes[i].text(y, x, cm[x, y], ha='center', va='center', color='black')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(cfg, history: dict, test_labels: np.ndarray,
                    test_preds: np.ndarray, test_probs: np.ndarray):
    """Generate all evaluation figures for an experiment."""
    figure_dir = os.path.join(cfg.output_dir, 'figures', cfg.experiment_name())
    os.makedirs(figure_dir, exist_ok=True)

    plot_loss_curves(history, os.path.join(figure_dir, 'loss_curves.png'))
    plot_f1_curve(
        history['val_f1_micro'],
        'Validation F1 (Micro) Over Epochs',
        os.path.join(figure_dir, 'f1_micro_curve.png'),
    )

    from evaluation.metrics import compute_per_class_f1
    f1_dict = compute_per_class_f1(test_labels, test_preds, cfg.labels)
    plot_per_class_f1(f1_dict, os.path.join(figure_dir, 'f1_per_class.png'))

    plot_pr_curves(test_labels, test_probs, cfg.labels,
                   os.path.join(figure_dir, 'pr_curves.png'))
    plot_confusion_matrices(test_labels, test_preds, cfg.labels,
                            os.path.join(figure_dir, 'confusion_matrices.png'))

    print(f"Figures saved to {figure_dir}/")
    return f1_dict


def plot_comparison_bar(results_summaries: list, save_path: str):
    """Cross-experiment bar chart comparing F1 micro and macro scores."""
    names = [r['experiment'] for r in results_summaries]
    f1_micro = [r['test_f1_micro'] for r in results_summaries]
    f1_macro = [r['test_f1_macro'] for r in results_summaries]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_micro, width, label='F1 Micro')
    bars2 = ax.bar(x + width/2, f1_macro, width, label='F1 Macro')

    ax.set_ylabel('F1 Score')
    ax.set_title('Cross-Experiment Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_comparison(results_summaries: list, labels: list, save_path: str):
    """Cross-experiment per-class F1 grouped bar chart."""
    n_classes = len(labels)
    n_exp = len(results_summaries)

    x = np.arange(n_classes)
    width = 0.8 / n_exp

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, r in enumerate(results_summaries):
        scores = [r['per_class_f1'].get(cls, 0) for cls in labels]
        offset = (i - (n_exp - 1) / 2) * width
        ax.bar(x + offset, scores, width, label=r['experiment'])

    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Comparison Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
