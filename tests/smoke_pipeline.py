"""End-to-end smoke test on synthetic data. Run as a script, not pytest.

Exercises: synthetic data → rectify → window → RF train → CNN train → predict.
The CNN runs 5 epochs only; goal is to prove the wiring, not to converge.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data import synthetic_subject
from src.evaluate import per_class_report
from src.features import extract_features
from src.models import EMG1DCNN
from src.preprocess import normalize_per_channel, rectify, window_signal
from src.train import TrainConfig, pick_device, predict, train_model


def main() -> None:
    print("device:", pick_device())

    all_w, all_y, all_r = [], [], []
    for s in (1, 2, 3):
        rec = synthetic_subject(subject=s, seed=s, n_classes=10,
                                reps_per_class=6, samples_per_rep=300)
        out = window_signal(rectify(rec.emg), rec.stimulus, rec.repetition,
                            fs=rec.fs, window_ms=200, overlap_ms=100)
        all_w.append(out["windows"])
        all_y.append(out["labels"])
        all_r.append(out["reps"])
    windows = np.concatenate(all_w)
    labels = np.concatenate(all_y)
    reps = np.concatenate(all_r)
    print("windows:", windows.shape, "classes:", np.unique(labels).size)

    tr = np.isin(reps, [1, 2, 3, 4])
    va = np.isin(reps, [5])
    te = np.isin(reps, [6])

    X_tr_f = extract_features(windows[tr])
    X_te_f = extract_features(windows[te])
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=0)
    rf.fit(X_tr_f, labels[tr])
    rf_pred = rf.predict(X_te_f)
    rf_acc = (rf_pred == labels[te]).mean()
    print(f"RF test acc on synthetic: {rf_acc:.3f}")

    X_tr, stats = normalize_per_channel(windows[tr])
    X_va, _ = normalize_per_channel(windows[va], stats=stats)
    X_te, _ = normalize_per_channel(windows[te], stats=stats)

    n_classes = int(labels.max()) + 1
    model = EMG1DCNN(n_channels=windows.shape[-1], n_classes=n_classes)
    cfg = TrainConfig(epochs=5, batch_size=32, lr=1e-3, patience=10)
    model, _ = train_model(model, (X_tr, labels[tr]), (X_va, labels[va]), cfg=cfg)

    cnn_pred = predict(model, X_te).argmax(axis=1)
    cnn_acc = (cnn_pred == labels[te]).mean()
    print(f"CNN test acc on synthetic (5 epochs): {cnn_acc:.3f}")

    rf_report = per_class_report(labels[te], rf_pred)
    cnn_report = per_class_report(labels[te], cnn_pred)
    print("RF macro F1:", rf_report["overall"]["macro_f1"])
    print("CNN macro F1:", cnn_report["overall"]["macro_f1"])
    print("smoke test OK")


if __name__ == "__main__":
    main()
