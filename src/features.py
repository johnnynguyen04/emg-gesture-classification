"""Time-domain features for the Random Forest baseline.

Six features per channel, ten channels per window → 60-dim vector. These are the
classical Hudgins TD set used in myoelectric control research.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("mav", "zc", "ssc", "wl", "rms", "var")


def mav(x: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(x), axis=1)


def zero_crossings(x: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    diff_sign = np.diff(np.sign(x), axis=1)
    mag = np.abs(np.diff(x, axis=1))
    return np.sum((diff_sign != 0) & (mag >= threshold), axis=1).astype(np.float32)


def slope_sign_changes(x: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    d = np.diff(x, axis=1)
    sign_change = np.diff(np.sign(d), axis=1) != 0
    mag_ok = (np.abs(d[:, 1:]) >= threshold) | (np.abs(d[:, :-1]) >= threshold)
    return np.sum(sign_change & mag_ok, axis=1).astype(np.float32)


def waveform_length(x: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(x, axis=1)), axis=1)


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(x ** 2, axis=1))


def variance(x: np.ndarray) -> np.ndarray:
    return np.var(x, axis=1)


def extract_features(windows: np.ndarray) -> np.ndarray:
    """windows: (n_windows, window_len, n_channels) → (n_windows, 6 * n_channels).

    Output columns are ordered (mav_ch0, mav_ch1, ..., var_ch9).
    """
    if windows.ndim != 3:
        raise ValueError(f"expected (n, T, C), got shape {windows.shape}")

    x = np.transpose(windows, (0, 2, 1)).reshape(-1, windows.shape[1])  # (n*C, T)
    feats = np.stack([mav(x), zero_crossings(x), slope_sign_changes(x),
                      waveform_length(x), rms(x), variance(x)], axis=1)  # (n*C, 6)
    n, _, c = windows.shape
    feats = feats.reshape(n, c, 6).transpose(0, 2, 1).reshape(n, 6 * c)
    return feats.astype(np.float32)


def feature_column_names(n_channels: int = 10) -> list[str]:
    return [f"{f}_ch{c}" for f in FEATURE_NAMES for c in range(n_channels)]
