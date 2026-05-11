"""NinaPro DB1 loader and synthetic stand-in for pipeline smoke tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# DB1 is sampled at 100 Hz on 10 Otto Bock electrodes, across three exercise files
# (E1, E2, E3) per subject. Stimulus 0 means rest; non-zero values are gesture IDs.
DB1_FS = 100
DB1_CHANNELS = 10
DB1_SUBJECTS = list(range(1, 28))
DB1_EXERCISES = (1, 2, 3)


@dataclass
class Recording:
    emg: np.ndarray         # (n_samples, n_channels)
    stimulus: np.ndarray    # (n_samples,) gesture id, 0 = rest
    repetition: np.ndarray  # (n_samples,) repetition index, 0 = rest
    fs: int
    subject: int


def _db1_filename(subject: int, exercise: int) -> str:
    return f"S{subject}_A1_E{exercise}.mat"


def load_subject(subject: int, data_dir: str | Path) -> Recording:
    """Load all three exercise files for one subject and concatenate them.

    Stimulus labels in E2 and E3 overlap with E1 (each file numbers from 1).
    We offset them so each gesture gets a unique class id across the full set:
    E1 keeps labels 1-12, E2 becomes 13-29, E3 becomes 30-52.
    """
    data_dir = Path(data_dir)
    offsets = {1: 0, 2: 12, 3: 29}

    emg_parts, stim_parts, rep_parts = [], [], []
    for ex in DB1_EXERCISES:
        path = data_dir / _db1_filename(subject, ex)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name} in {data_dir}. "
                f"Download DB1 from ninapro.hevs.ch and place all S{subject}_A1_E*.mat files there."
            )
        mat = loadmat(str(path))
        emg = np.asarray(mat["emg"], dtype=np.float32)
        stim = np.asarray(mat["restimulus"], dtype=np.int64).ravel()
        rep = np.asarray(mat["rerepetition"], dtype=np.int64).ravel()
        stim = np.where(stim > 0, stim + offsets[ex], 0)
        emg_parts.append(emg)
        stim_parts.append(stim)
        rep_parts.append(rep)

    return Recording(
        emg=np.concatenate(emg_parts, axis=0),
        stimulus=np.concatenate(stim_parts, axis=0),
        repetition=np.concatenate(rep_parts, axis=0),
        fs=DB1_FS,
        subject=subject,
    )


def synthetic_subject(subject: int = 1, seed: int | None = None, n_classes: int = 53,
                      reps_per_class: int = 6, samples_per_rep: int = 500) -> Recording:
    """Generate a fake recording with class-dependent structure.

    Each gesture imprints a distinct amplitude + frequency pattern on the channels,
    so a properly wired pipeline should learn it. Used for end-to-end smoke tests
    when the real DB1 files are not available yet.
    """
    rng = np.random.default_rng(seed if seed is not None else subject)
    total_samples = n_classes * reps_per_class * samples_per_rep
    emg = rng.normal(0, 0.01, size=(total_samples, DB1_CHANNELS)).astype(np.float32)
    stimulus = np.zeros(total_samples, dtype=np.int64)
    repetition = np.zeros(total_samples, dtype=np.int64)

    t = np.arange(samples_per_rep) / DB1_FS
    idx = 0
    for cls in range(n_classes):
        chan_amps = rng.uniform(0.05, 0.25, size=DB1_CHANNELS)
        chan_freqs = rng.uniform(2, 20, size=DB1_CHANNELS)
        for rep in range(1, reps_per_class + 1):
            for ch in range(DB1_CHANNELS):
                emg[idx:idx + samples_per_rep, ch] += (
                    chan_amps[ch] * np.sin(2 * np.pi * chan_freqs[ch] * t)
                )
            stimulus[idx:idx + samples_per_rep] = cls if cls > 0 else 0
            repetition[idx:idx + samples_per_rep] = rep
            idx += samples_per_rep

    return Recording(emg=emg, stimulus=stimulus, repetition=repetition,
                     fs=DB1_FS, subject=subject)


def has_real_db1(data_dir: str | Path, subject: int = 1) -> bool:
    data_dir = Path(data_dir)
    return all((data_dir / _db1_filename(subject, ex)).exists() for ex in DB1_EXERCISES)


def load_or_synthesize(subject: int, data_dir: str | Path,
                       seed: int | None = None) -> Recording:
    """Real DB1 if present, otherwise a deterministic synthetic stand-in."""
    if has_real_db1(data_dir, subject):
        return load_subject(subject, data_dir)
    return synthetic_subject(subject=subject, seed=seed)
