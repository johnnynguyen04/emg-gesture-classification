"""Regenerate RF and CNN confusion matrix PNGs with the demo's blue palette.

Uses the existing windows.npz, retrains the RF baseline (fast) and runs CNN
inference from the saved checkpoint. Saves to results/figures/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from src.features import extract_features
from src.models import EMG1DCNN
from src.preprocess import normalize_per_channel
from src.train import pick_device, predict

PROC = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
FIGS = ROOT / "results" / "figures"

# Match the Streamlit demo palette.
PRIMARY = "#1ba8e1"
TEXT = "#0c2d42"
MUTED = "#6e8296"
BG = "#fcfdfe"
BORDER = "#dce7ef"

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "demo_blue", ["#fcfdfe", "#e0f2fb", "#8bd0ef", PRIMARY, "#005c95"]
)

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Plus Jakarta Sans", "Segoe UI", "Arial", "sans-serif"],
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "axes.titlesize": 11,
    "axes.titleweight": "regular",
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize": 10,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
}


def plot_cm(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_n = cm.astype(float) / row_sums

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.0, 5.4))
        sns.heatmap(
            cm_n, ax=ax, cmap=BLUE_CMAP, square=True,
            cbar=True, cbar_kws={"shrink": 0.7, "label": "row-normalized rate"},
            xticklabels=False, yticklabels=False, vmin=0, vmax=1,
        )
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(title, pad=12)
        cbar = ax.collections[0].colorbar
        cbar.outline.set_edgecolor(BORDER)
        cbar.ax.tick_params(length=0, labelsize=8, colors=MUTED)
        cbar.ax.yaxis.label.set_color(MUTED)
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    print(f"  saved {save_path.name}")


def main() -> int:
    print("loading windows.npz")
    d = np.load(PROC / "windows.npz")
    windows, labels, reps = d["windows"], d["labels"], d["reps"]

    tr = np.isin(reps, [1, 2, 3, 4, 5, 6, 7])
    te = np.isin(reps, [9, 10])
    y_te = labels[te]

    print("retraining RF")
    t0 = time.time()
    X_tr = extract_features(windows[tr])
    X_te = extract_features(windows[te])
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    rf.fit(X_tr, labels[tr])
    rf_pred = rf.predict(X_te)
    print(f"  RF retrained in {time.time() - t0:.1f}s")
    plot_cm(y_te, rf_pred, "Random Forest — DB1 test", FIGS / "rf_confusion.png")

    print("running CNN inference")
    device = pick_device()
    X_te_n, _ = normalize_per_channel(
        windows[te],
        stats={
            "mean": np.load(METRICS / "norm_stats.npz")["mean"],
            "std": np.load(METRICS / "norm_stats.npz")["std"],
        },
    )
    model = EMG1DCNN(n_channels=10, n_classes=53)
    model.load_state_dict(torch.load(METRICS / "cnn_best.pt", map_location="cpu"))
    probs = predict(model, X_te_n, device=device)
    cnn_pred = probs.argmax(axis=1)
    plot_cm(y_te, cnn_pred, "1D CNN — DB1 test", FIGS / "cnn_confusion.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
