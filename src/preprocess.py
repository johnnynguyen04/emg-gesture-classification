"""Bandpass, rectify, and windowing for sEMG signals.

DB1 caveat: the raw signal is already a rectified RMS envelope from the Otto Bock
electrodes, sampled at 100 Hz. Applying a 20-450 Hz bandpass is meaningless here
(Nyquist = 50 Hz). The functions exist for portability to DB2/DB5 (raw EMG at 2 kHz)
and are skipped in the default DB1 pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(signal: np.ndarray, fs: int, low: float = 20.0,
                    high: float = 450.0, order: int = 4) -> np.ndarray:
    """4th-order zero-phase Butterworth bandpass. Skip for DB1 (fs too low)."""
    nyq = fs / 2
    if high >= nyq:
        raise ValueError(
            f"Bandpass high cutoff {high} Hz exceeds Nyquist ({nyq} Hz) at fs={fs}. "
            f"DB1 is 100 Hz and pre-rectified — skip this step."
        )
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal, axis=0).astype(np.float32)


def rectify(signal: np.ndarray) -> np.ndarray:
    return np.abs(signal).astype(np.float32)


def window_signal(emg: np.ndarray, stimulus: np.ndarray, repetition: np.ndarray,
                  fs: int, window_ms: int = 200, overlap_ms: int = 100,
                  drop_rest: bool = False) -> dict[str, np.ndarray]:
    """Slide fixed-length windows over the signal. Label by stimulus at window center.

    A window is discarded only if its stimulus is mixed (gesture transition mid-window)
    so we never train on ambiguous labels. Returns dict with arrays:
      windows:    (n_windows, window_len, n_channels)
      labels:     (n_windows,)
      reps:       (n_windows,) repetition id of the window
    """
    window_len = int(round(fs * window_ms / 1000))
    step = int(round(fs * (window_ms - overlap_ms) / 1000))
    if step < 1:
        raise ValueError("overlap_ms must be smaller than window_ms")

    n_samples = emg.shape[0]
    n_windows = 1 + (n_samples - window_len) // step

    windows, labels, reps = [], [], []
    for i in range(n_windows):
        start = i * step
        end = start + window_len
        seg_stim = stimulus[start:end]
        if seg_stim[0] != seg_stim[-1]:
            continue  # crossing a gesture boundary
        label = int(seg_stim[window_len // 2])
        if drop_rest and label == 0:
            continue
        windows.append(emg[start:end])
        labels.append(label)
        reps.append(int(repetition[start:end][window_len // 2]))

    return {
        "windows": np.stack(windows).astype(np.float32) if windows else np.empty((0, window_len, emg.shape[1]), dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "reps": np.asarray(reps, dtype=np.int64),
    }


def normalize_per_channel(windows: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Z-score each channel using train-set statistics.

    Pass stats=None for the train split, then reuse the returned dict on val/test.
    """
    if stats is None:
        # flatten across windows × time, keep channel axis
        flat = windows.reshape(-1, windows.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-8
        stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    normed = (windows - stats["mean"]) / stats["std"]
    return normed.astype(np.float32), stats
