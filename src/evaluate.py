"""Metrics, confusion matrix plotting, and a comparison helper for RF vs CNN."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None,
                     target_names: list[str] | None = None) -> dict:
    """Wrap sklearn's classification_report and add macro/weighted F1 explicitly."""
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    report["overall"] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return report


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None,
                          title: str = "Confusion matrix", save_path: str | Path | None = None,
                          normalize: bool = True) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, ax=ax, cmap="viridis", square=True, cbar=True,
                xticklabels=False, yticklabels=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)


def compare_models(rf_metrics: dict, cnn_metrics: dict) -> dict:
    return {
        "random_forest": {
            "accuracy": rf_metrics["overall"]["accuracy"],
            "macro_f1": rf_metrics["overall"]["macro_f1"],
            "weighted_f1": rf_metrics["overall"]["weighted_f1"],
        },
        "cnn_1d": {
            "accuracy": cnn_metrics["overall"]["accuracy"],
            "macro_f1": cnn_metrics["overall"]["macro_f1"],
            "weighted_f1": cnn_metrics["overall"]["weighted_f1"],
        },
        "delta_accuracy": cnn_metrics["overall"]["accuracy"] - rf_metrics["overall"]["accuracy"],
    }


def hardest_classes(y_true: np.ndarray, y_pred: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
    """Return classes with the lowest per-class recall (most confused targets)."""
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recall = float((y_pred[mask] == c).mean())
        recalls.append((int(c), recall))
    recalls.sort(key=lambda kv: kv[1])
    return recalls[:top_k]
