"""Resume the pipeline at the CNN stage, using the existing data/processed/windows.npz.

Uses a larger batch size and fewer epochs than the original run because we
already know the dataset is 1.2M windows and CPU training was the bottleneck.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.evaluate import (
    compare_models,
    hardest_classes,
    per_class_report,
    plot_confusion_matrix,
    save_metrics,
)
from src.models import EMG1DCNN, count_parameters
from src.preprocess import normalize_per_channel
from src.train import TrainConfig, pick_device, predict, train_model

PROC = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
FIGS = ROOT / "results" / "figures"
SAMPLES = ROOT / "streamlit_app" / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("loading windows.npz")
    d = np.load(PROC / "windows.npz")
    windows, labels, reps = d["windows"], d["labels"], d["reps"]
    print(f"  windows {windows.shape}, classes {np.unique(labels).size}")

    device = pick_device()
    print(f"device {device}")

    tr = np.isin(reps, [1, 2, 3, 4, 5, 6, 7])
    va = np.isin(reps, [8])
    te = np.isin(reps, [9, 10])

    X_tr, stats = normalize_per_channel(windows[tr])
    X_va, _ = normalize_per_channel(windows[va], stats=stats)
    X_te, _ = normalize_per_channel(windows[te], stats=stats)
    y_tr, y_va, y_te = labels[tr], labels[va], labels[te]
    print(f"train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    n_classes = int(labels.max()) + 1
    model = EMG1DCNN(n_channels=X_tr.shape[-1], n_classes=n_classes)
    print(f"params {count_parameters(model):,}, classes {n_classes}")

    cfg = TrainConfig(
        epochs=30, batch_size=512, lr=2e-3, weight_decay=1e-4, patience=6,
        checkpoint_path=str(METRICS / "cnn_best.pt"),
    )
    t0 = time.time()
    model, history = train_model(model, (X_tr, y_tr), (X_va, y_va), cfg=cfg, device=device)
    np.savez(METRICS / "norm_stats.npz", **stats)
    print(f"CNN training took {time.time() - t0:.1f}s")

    probs = predict(model, X_te, device=device)
    y_pred = probs.argmax(axis=1)
    cnn_report = per_class_report(y_te, y_pred)
    save_metrics(cnn_report, METRICS / "cnn_metrics.json")
    plot_confusion_matrix(y_te, y_pred,
                          title="1D CNN — DB1 test confusion matrix",
                          save_path=FIGS / "cnn_confusion.png")

    rf_report = json.loads((METRICS / "rf_metrics.json").read_text())
    comparison = compare_models(rf_report, cnn_report)
    save_metrics(comparison, METRICS / "comparison.json")

    label_map = {int(c): f"class_{int(c):02d}" for c in range(n_classes)}
    (METRICS / "label_map.json").write_text(json.dumps(label_map, indent=2))
    for cls in (0, 1, 5, 12, 25, 40):
        idxs = np.where(y_te == cls)[0]
        if len(idxs):
            np.save(SAMPLES / f"sample_class{cls:02d}.npy",
                    X_te[idxs[0]] * stats["std"] + stats["mean"])

    print("\n=== summary ===")
    print(f"RF   acc {comparison['random_forest']['accuracy']:.4f}  "
          f"macro F1 {comparison['random_forest']['macro_f1']:.4f}")
    print(f"CNN  acc {comparison['cnn_1d']['accuracy']:.4f}  "
          f"macro F1 {comparison['cnn_1d']['macro_f1']:.4f}")
    print(f"delta {comparison['delta_accuracy']:+.4f}")
    print("CNN hardest classes (class, recall):", hardest_classes(y_te, y_pred, top_k=5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
