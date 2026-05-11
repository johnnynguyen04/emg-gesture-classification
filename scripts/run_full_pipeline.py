"""Run the full DB1 within-subject pipeline end to end.

Stages:
  1. Load all 27 subjects, rectify, window → data/processed/windows.npz
  2. Train RF on Hudgins features → results/metrics/rf_*.json + figure
  3. Train CNN on normalized windows → results/metrics/cnn_*.{pt,json} + figure
  4. Save sample test windows for the Streamlit demo
  5. Print a comparison summary

Repetition-based split: reps 1-7 train, 8 val, 9-10 test (70/10/20).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.data import DB1_SUBJECTS, has_real_db1, load_subject
from src.evaluate import (
    compare_models,
    hardest_classes,
    per_class_report,
    plot_confusion_matrix,
    save_metrics,
)
from src.features import extract_features
from src.models import EMG1DCNN, count_parameters
from src.preprocess import normalize_per_channel, rectify, window_signal
from src.train import TrainConfig, pick_device, predict, train_model

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
FIGS = ROOT / "results" / "figures"
SAMPLES = ROOT / "streamlit_app" / "samples"

for p in (PROC, METRICS, FIGS, SAMPLES):
    p.mkdir(parents=True, exist_ok=True)


def stage_preprocess() -> dict:
    print(f"[1/4] preprocessing {len(DB1_SUBJECTS)} subjects")
    t0 = time.time()
    parts_w, parts_y, parts_r, parts_s = [], [], [], []
    for s in tqdm(DB1_SUBJECTS):
        if not has_real_db1(RAW, subject=s):
            print(f"  skipping S{s} (files missing)")
            continue
        rec = load_subject(s, RAW)
        out = window_signal(
            rectify(rec.emg), rec.stimulus, rec.repetition,
            fs=rec.fs, window_ms=200, overlap_ms=100,
        )
        if out["windows"].size == 0:
            continue
        parts_w.append(out["windows"])
        parts_y.append(out["labels"])
        parts_r.append(out["reps"])
        parts_s.append(np.full(out["labels"].shape[0], s, dtype=np.int64))

    windows = np.concatenate(parts_w)
    labels = np.concatenate(parts_y)
    reps = np.concatenate(parts_r)
    subjects = np.concatenate(parts_s)
    np.savez_compressed(PROC / "windows.npz",
                        windows=windows, labels=labels, reps=reps, subjects=subjects)
    print(f"  windows {windows.shape}, classes {np.unique(labels).size}, "
          f"took {time.time() - t0:.1f}s")
    return {"windows": windows, "labels": labels, "reps": reps, "subjects": subjects}


def stage_rf(d: dict) -> dict:
    print("[2/4] random forest")
    t0 = time.time()
    tr = np.isin(d["reps"], [1, 2, 3, 4, 5, 6, 7])
    te = np.isin(d["reps"], [9, 10])

    X_tr = extract_features(d["windows"][tr])
    X_te = extract_features(d["windows"][te])
    y_tr = d["labels"][tr]
    y_te = d["labels"][te]
    print(f"  feature matrix train {X_tr.shape}, test {X_te.shape}")

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    report = per_class_report(y_te, y_pred)
    save_metrics(report, METRICS / "rf_metrics.json")
    plot_confusion_matrix(y_te, y_pred,
                          title="Random Forest — DB1 test confusion matrix",
                          save_path=FIGS / "rf_confusion.png")
    print(f"  RF test acc {report['overall']['accuracy']:.4f}, "
          f"macro F1 {report['overall']['macro_f1']:.4f}, "
          f"took {time.time() - t0:.1f}s")
    return report


def stage_cnn(d: dict) -> dict:
    print("[3/4] 1D CNN")
    t0 = time.time()
    device = pick_device()
    print(f"  device {device}")

    tr = np.isin(d["reps"], [1, 2, 3, 4, 5, 6, 7])
    va = np.isin(d["reps"], [8])
    te = np.isin(d["reps"], [9, 10])

    X_tr, stats = normalize_per_channel(d["windows"][tr])
    X_va, _ = normalize_per_channel(d["windows"][va], stats=stats)
    X_te, _ = normalize_per_channel(d["windows"][te], stats=stats)
    y_tr, y_va, y_te = d["labels"][tr], d["labels"][va], d["labels"][te]
    print(f"  train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    n_classes = int(d["labels"].max()) + 1
    model = EMG1DCNN(n_channels=X_tr.shape[-1], n_classes=n_classes)
    print(f"  params {count_parameters(model):,}, classes {n_classes}")

    cfg = TrainConfig(
        epochs=50, batch_size=128, lr=1e-3, weight_decay=1e-4, patience=8,
        checkpoint_path=str(METRICS / "cnn_best.pt"),
    )
    model, history = train_model(model, (X_tr, y_tr), (X_va, y_va), cfg=cfg, device=device)
    np.savez(METRICS / "norm_stats.npz", **stats)

    probs = predict(model, X_te, device=device)
    y_pred = probs.argmax(axis=1)
    report = per_class_report(y_te, y_pred)
    save_metrics(report, METRICS / "cnn_metrics.json")
    plot_confusion_matrix(y_te, y_pred,
                          title="1D CNN — DB1 test confusion matrix",
                          save_path=FIGS / "cnn_confusion.png")
    print(f"  CNN test acc {report['overall']['accuracy']:.4f}, "
          f"macro F1 {report['overall']['macro_f1']:.4f}, "
          f"took {time.time() - t0:.1f}s")
    return report, model, stats, X_te, y_te, y_pred


def stage_samples(model, stats, X_te, y_te) -> None:
    print("[4/4] saving demo samples")
    n_classes = int(y_te.max()) + 1
    label_map = {int(c): f"class_{int(c):02d}" for c in range(n_classes)}
    (METRICS / "label_map.json").write_text(json.dumps(label_map, indent=2))

    chosen = []
    for cls in (0, 1, 5, 12, 25, 40):
        idxs = np.where(y_te == cls)[0]
        if len(idxs):
            chosen.append((cls, int(idxs[0])))
    for cls, i in chosen:
        np.save(SAMPLES / f"sample_class{cls:02d}.npy", X_te[i] * stats["std"] + stats["mean"])
    print(f"  wrote {len(chosen)} sample windows + label map")


def main() -> int:
    data = stage_preprocess()
    rf_report = stage_rf(data)
    cnn_report, model, stats, X_te, y_te, cnn_pred = stage_cnn(data)
    stage_samples(model, stats, X_te, y_te)

    comparison = compare_models(rf_report, cnn_report)
    save_metrics(comparison, METRICS / "comparison.json")

    print("\n=== summary ===")
    print(f"RF   acc {comparison['random_forest']['accuracy']:.4f}  "
          f"macro F1 {comparison['random_forest']['macro_f1']:.4f}")
    print(f"CNN  acc {comparison['cnn_1d']['accuracy']:.4f}  "
          f"macro F1 {comparison['cnn_1d']['macro_f1']:.4f}")
    print(f"delta {comparison['delta_accuracy']:+.4f}")
    print("CNN hardest classes (class, recall):", hardest_classes(y_te, cnn_pred, top_k=5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
