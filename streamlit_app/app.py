"""Streamlit demo: upload a 200 ms × 10ch EMG window and see the predicted gesture."""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import EMG1DCNN # noqa: E402
from src.preprocess import normalize_per_channel # noqa: E402

CHECKPOINT = ROOT / "results" / "metrics" / "cnn_best.pt"
NORM_STATS = ROOT / "results" / "metrics" / "norm_stats.npz"
LABEL_MAP = ROOT / "results" / "metrics" / "label_map.json"
SAMPLES_DIR = ROOT / "streamlit_app" / "samples"


def gesture_group(class_id: int) -> str:
    if class_id == 0:
        return "rest"
    if 1 <= class_id <= 12:
        return "basic finger movement"
    if 13 <= class_id <= 29:
        return "hand configuration"
    return "functional grasp"


@st.cache_resource
def load_model() -> tuple[EMG1DCNN, dict, dict]:
    if not CHECKPOINT.exists():
        return None, None, None
    model = EMG1DCNN(n_channels=10, n_classes=53)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    stats = {}
    if NORM_STATS.exists():
        npz = np.load(NORM_STATS)
        stats = {"mean": npz["mean"], "std": npz["std"]}
    labels = {}
    if LABEL_MAP.exists():
        labels = {int(k): v for k, v in json.loads(LABEL_MAP.read_text()).items()}
    return model, stats, labels


def plot_window(window: np.ndarray) -> plt.Figure:
    n_t, n_c = window.shape
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(n_t) / 100
    for c in range(n_c):
        ax.plot(t, window[:, c] + 0.5 * c, lw=0.8, label=f"ch{c}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("forearm channels (stacked)")
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def predict(model: EMG1DCNN, window: np.ndarray, stats: dict) -> np.ndarray:
    if stats:
        window, _ = normalize_per_channel(window[None, ...], stats=stats)
    else:
        window = window[None, ...]
    x = torch.from_numpy(np.transpose(window, (0, 2, 1))).float()
    with torch.no_grad():
        probs = model(x).softmax(dim=1).numpy()[0]
    return probs


def true_class_from_filename(name: str) -> int | None:
    m = re.search(r"class(\d+)", name)
    return int(m.group(1)) if m else None


def render_intro(labels: dict) -> None:
    st.markdown(
        "Surface EMG records electrical activity from forearm muscles while the "
        "hand moves. This demo takes a **200 ms snapshot** of EMG (10 forearm "
        "channels, 100 Hz) and predicts which of **53 hand gestures** the person "
        "is doing, using a 1D CNN trained on the public **NinaPro DB1** dataset."
    )
    st.markdown(
        "The CNN reaches **35% top-1 accuracy** on held-out repetitions — modest, "
        "because the model is small (60k parameters) and trained without "
        "augmentation. The Random Forest baseline reached 53% on the same split. "
        "See the README in the repo for the full writeup."
    )

    with st.expander("What are the 53 gestures?"):
        groups = [
            ("Rest (class 0)", [0]),
            ("Basic finger movements (classes 1–12)", list(range(1, 13))),
            ("Hand configurations (classes 13–29)", list(range(13, 30))),
            ("Functional grasps (classes 30–52)", list(range(30, 53))),
        ]
        for title, ids in groups:
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- `{i}` — {labels.get(i, '?')}" for i in ids))

    with st.expander("How to read this page"):
        st.markdown(
            "- **Input signal**: ten lines, one per electrode placed around the "
            "forearm, stacked vertically. The signal is already a rectified "
            "envelope from the Otto Bock electrodes used in DB1, so it looks "
            "smoother than raw EMG.\n"
            "- **Prediction**: the gesture the model thinks is happening, plus its "
            "confidence and which gesture group it belongs to.\n"
            "- **Top-5**: the model's five most likely guesses with probability. "
            "Low max probability = the model knows it's unsure."
        )


def main() -> None:
    st.set_page_config(page_title="EMG gesture classifier", layout="centered")
    st.title("EMG gesture classifier")

    model, stats, labels = load_model()
    if model is None:
        st.warning(
            f"No trained model found at `{CHECKPOINT.relative_to(ROOT)}`. "
            f"Train the CNN first (notebook 04) and rerun the app."
        )
        st.stop()

    render_intro(labels)
    st.divider()

    samples = sorted(SAMPLES_DIR.glob("*.npy")) if SAMPLES_DIR.exists() else []
    source = st.radio("Input source", ["Pick a sample", "Upload .npy"], horizontal=True,
                      disabled=not samples)

    window = None
    true_id = None
    if source == "Upload .npy":
        uploaded = st.file_uploader("Upload a (T, 10) float32 array", type=["npy"])
        if uploaded is not None:
            window = np.load(io.BytesIO(uploaded.read()))
    else:
        sample_labels = []
        for s in samples:
            tid = true_class_from_filename(s.name)
            name = labels.get(tid, s.name) if tid is not None else s.name
            sample_labels.append(f"{name} (true class {tid})")
        idx = st.selectbox("Sample (real DB1 test windows)",
                           range(len(samples)),
                           format_func=lambda i: sample_labels[i])
        if idx is not None:
            window = np.load(samples[idx])
            true_id = true_class_from_filename(samples[idx].name)

    if window is None:
        st.info("Waiting for input.")
        return

    if window.shape[1] != 10:
        st.error(f"Expected 10 channels, got shape {window.shape}.")
        return

    st.subheader("Input signal")
    st.pyplot(plot_window(window))

    probs = predict(model, window.astype(np.float32), stats)
    top_idx = probs.argsort()[::-1][:5]
    top_probs = probs[top_idx]
    top_names = [labels.get(int(i), f"class_{int(i)}") for i in top_idx]

    pred_id = int(top_idx[0])
    pred_name = top_names[0]
    pred_group = gesture_group(pred_id)
    st.subheader(f"Prediction: {pred_name}")
    st.markdown(
        f"**Confidence:** {top_probs[0]:.1%} · **Group:** {pred_group} · "
        f"**Class id:** {pred_id}"
    )

    if true_id is not None:
        true_name = labels.get(true_id, f"class_{true_id}")
        if true_id == pred_id:
            st.success(f"Correct — true label is **{true_name}**.")
        elif true_id in top_idx.tolist():
            rank = top_idx.tolist().index(true_id) + 1
            st.warning(
                f"Top-1 wrong, but true label **{true_name}** is rank {rank} "
                f"({probs[true_id]:.1%})."
            )
        else:
            st.error(
                f"Missed. True label **{true_name}** not in top 5 "
                f"(model gave it {probs[true_id]:.1%})."
            )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(range(len(top_idx)), top_probs[::-1])
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("probability")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
