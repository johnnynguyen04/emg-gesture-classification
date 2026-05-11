import numpy as np
import torch

from src.data import DB1_CHANNELS, DB1_FS, synthetic_subject
from src.features import extract_features, feature_column_names
from src.models import EMG1DCNN
from src.preprocess import normalize_per_channel, rectify, window_signal


def test_synthetic_recording_shape():
    rec = synthetic_subject(subject=1, seed=0, n_classes=10, reps_per_class=3, samples_per_rep=200)
    assert rec.emg.shape[1] == DB1_CHANNELS
    assert rec.emg.shape[0] == rec.stimulus.shape[0] == rec.repetition.shape[0]
    assert rec.fs == DB1_FS


def test_windowing_produces_labelled_windows():
    rec = synthetic_subject(subject=2, seed=0, n_classes=10, reps_per_class=3, samples_per_rep=200)
    out = window_signal(rec.emg, rec.stimulus, rec.repetition, fs=rec.fs,
                        window_ms=200, overlap_ms=100)
    assert out["windows"].ndim == 3
    assert out["windows"].shape[1:] == (20, DB1_CHANNELS)
    assert out["windows"].shape[0] == out["labels"].shape[0]


def test_features_shape_and_finite():
    rec = synthetic_subject(subject=3, seed=0, n_classes=5, reps_per_class=2, samples_per_rep=200)
    out = window_signal(rectify(rec.emg), rec.stimulus, rec.repetition, fs=rec.fs)
    feats = extract_features(out["windows"])
    assert feats.shape == (out["windows"].shape[0], 60)
    assert np.isfinite(feats).all()
    assert len(feature_column_names()) == 60


def test_normalize_reuses_train_stats():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, size=(50, 20, 10)).astype(np.float32)
    val = rng.normal(0, 1, size=(20, 20, 10)).astype(np.float32)
    train_n, stats = normalize_per_channel(train)
    val_n, _ = normalize_per_channel(val, stats=stats)
    assert train_n.shape == train.shape and val_n.shape == val.shape


def test_cnn_forward_pass():
    model = EMG1DCNN(n_channels=10, n_classes=53)
    dummy = torch.randn(8, 10, 20)
    out = model(dummy)
    assert out.shape == (8, 53)
    assert torch.isfinite(out).all()
