import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> dict:
    f1_per_class = f1_score(y_true, y_pred, average=None)
    return dict(zip(labels, f1_per_class))


def compute_precision_recall_curves(y_true: np.ndarray, y_probs: np.ndarray, labels: list) -> dict:
    curves = {}
    for i, label in enumerate(labels):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        curves[label] = {'precision': precision, 'recall': recall, 'thresholds': thresholds}
    return curves


def compute_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> dict:
    matrices = {}
    for i, label in enumerate(labels):
        matrices[label] = confusion_matrix(y_true[:, i], y_pred[:, i])
    return matrices
