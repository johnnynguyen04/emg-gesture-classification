"""Streamlit demo: predict a hand gesture from a 200 ms × 10ch sEMG window."""
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
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import EMG1DCNN # noqa: E402
from src.preprocess import normalize_per_channel # noqa: E402

CHECKPOINT = ROOT / "results" / "metrics" / "cnn_best.pt"
NORM_STATS = ROOT / "results" / "metrics" / "norm_stats.npz"
LABEL_MAP = ROOT / "results" / "metrics" / "label_map.json"
COMPARISON = ROOT / "results" / "metrics" / "comparison.json"
SAMPLES_DIR = ROOT / "streamlit_app" / "samples"

PRIMARY = "#0891b2"
PRIMARY_DARK = "#0e7490"
ACCENT = "#059669"
TEXT = "#18181b"
MUTED = "#71717a"
BORDER = "#e4e4e7"
BG = "#fafafa"
CHANNEL_CMAP = LinearSegmentedColormap.from_list(
    "channels", ["#a5f3fc", "#0891b2", "#0c4a6e"]
)

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Geist", "Inter", "Helvetica", "Arial", "sans-serif"],
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "regular",
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
}


def gesture_group(class_id: int) -> str:
    if class_id == 0:
        return "rest"
    if 1 <= class_id <= 12:
        return "basic finger movement"
    if 13 <= class_id <= 29:
        return "hand configuration"
    return "functional grasp"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');

        html, body, [class*="st-"], button, input, textarea, select {
            font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        h1, h2, h3, h4 {
            font-family: 'Geist', sans-serif !important;
            letter-spacing: -0.02em;
            color: #18181b;
        }
        h1 { font-weight: 600; }
        [data-testid="stMetricValue"] {
            font-family: 'Geist Mono', monospace !important;
            font-weight: 500;
            color: #18181b;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #71717a;
        }
        [data-testid="stSidebar"] { border-right: 1px solid #e4e4e7; }
        .stButton button, .stDownloadButton button { border-radius: 6px; }
        hr { border-color: #e4e4e7; }
        .small-muted { color: #71717a; font-size: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model() -> tuple[EMG1DCNN | None, dict, dict, dict]:
    if not CHECKPOINT.exists():
        return None, {}, {}, {}
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
    comparison = {}
    if COMPARISON.exists():
        comparison = json.loads(COMPARISON.read_text())
    return model, stats, labels, comparison


def plot_signal(window: np.ndarray) -> plt.Figure:
    n_t, n_c = window.shape
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        t = np.arange(n_t) * 1000 / 100 # ms
        offsets = np.arange(n_c) * 0.55
        colors = [CHANNEL_CMAP(i / max(n_c - 1, 1)) for i in range(n_c)]
        for c in range(n_c):
            ax.plot(t, window[:, c] + offsets[c], lw=1.1, color=colors[c])
        ax.set_xlabel("time (ms)")
        ax.set_yticks(offsets)
        ax.set_yticklabels([f"ch {c + 1}" for c in range(n_c)])
        ax.set_xlim(0, t[-1])
        ax.tick_params(length=3)
        fig.tight_layout()
    return fig


def plot_top5(top_probs: np.ndarray, top_names: list[str], correct_idx: int | None) -> plt.Figure:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        colors = [PRIMARY] * len(top_probs)
        if correct_idx is not None:
            colors[correct_idx] = ACCENT
        ax.barh(range(len(top_probs)), top_probs[::-1], color=colors[::-1], height=0.65)
        ax.set_yticks(range(len(top_probs)))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability")
        for i, p in enumerate(top_probs[::-1]):
            ax.text(p + 0.012, i, f"{p:.1%}", va="center", fontsize=9, color=TEXT)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ax.tick_params(axis="y", length=0)
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


def render_sidebar(comparison: dict) -> None:
    with st.sidebar:
        st.markdown("##### Model card")
        st.markdown(
            '<span class="small-muted">A 1D convolutional network trained on '
            "NinaPro DB1, a public sEMG gesture dataset.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        cnn = comparison.get("cnn_1d", {})
        rf = comparison.get("random_forest", {})

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Subjects", "27")
            st.metric("CNN params", "60k")
            st.metric("CNN test acc",
                      f"{cnn.get('accuracy', 0):.1%}" if cnn else "—")
        with c2:
            st.metric("Classes", "53")
            st.metric("Train windows", "350k")
            st.metric("RF test acc",
                      f"{rf.get('accuracy', 0):.1%}" if rf else "—")

        st.divider()
        st.markdown("##### Honest take")
        st.markdown(
            '<span class="small-muted">The Random Forest baseline beats this '
            "small CNN by 18 points. DB1's 100 Hz pre-rectified envelope leaves "
            "little for a deep model to extract. A larger CNN with augmentation "
            "is the v2 plan; see the README on GitHub for the full writeup."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.link_button(
            "Repo on GitHub",
            "https://github.com/johnnynguyen04/emg-gesture-classification",
            use_container_width=True,
        )


def render_hero() -> None:
    st.markdown(
        "<h1 style='margin-bottom: 0.25rem;'>EMG gesture classifier</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="small-muted" style="margin-top:0; font-size: 1rem;">'
        "Pick a 200 ms snapshot of forearm muscle activity. The model predicts "
        "which of 53 hand gestures the person was performing."
        "</p>",
        unsafe_allow_html=True,
    )


def render_intro_expanders(labels: dict) -> None:
    with st.expander("What are the 53 gestures?"):
        groups = [
            ("Rest (class 0)", [0]),
            ("Basic finger movements · classes 1–12", list(range(1, 13))),
            ("Hand configurations · classes 13–29", list(range(13, 30))),
            ("Functional grasps · classes 30–52", list(range(30, 53))),
        ]
        for title, ids in groups:
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- `{i}` · {labels.get(i, '?')}" for i in ids))

    with st.expander("How to read this page"):
        st.markdown(
            "- **Input signal**: ten traces, one per forearm electrode, stacked "
            "vertically. NinaPro DB1 uses Otto Bock electrodes that rectify and "
            "smooth the signal in hardware, so it looks calmer than raw EMG.\n"
            "- **Prediction**: the gesture the model is betting on, with its "
            "confidence and which group it belongs to.\n"
            "- **Top-5**: the model's five most likely guesses with probability. "
            "Low max probability means the model knows it is unsure. The bar in "
            "green (if any) marks the true label."
        )


def render_demo(model, stats, labels) -> None:
    samples = sorted(SAMPLES_DIR.glob("*.npy")) if SAMPLES_DIR.exists() else []

    source = st.radio(
        "Input",
        ["Pick a sample", "Upload .npy"],
        horizontal=True,
        label_visibility="collapsed",
        disabled=not samples,
    )

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
            sample_labels.append(f"{name} · true class {tid}")
        idx = st.selectbox(
            "Real test windows from DB1",
            range(len(samples)),
            format_func=lambda i: sample_labels[i],
        )
        if idx is not None:
            window = np.load(samples[idx])
            true_id = true_class_from_filename(samples[idx].name)

    if window is None:
        st.info("Pick or upload a window to run the model.")
        return

    if window.shape[1] != 10:
        st.error(f"Expected 10 channels, got shape {window.shape}.")
        return

    probs = predict(model, window.astype(np.float32), stats)
    top_idx = probs.argsort()[::-1][:5]
    top_probs = probs[top_idx]
    top_names = [labels.get(int(i), f"class_{int(i)}") for i in top_idx]
    pred_id = int(top_idx[0])
    pred_name = top_names[0]

    st.markdown("&nbsp;", unsafe_allow_html=True)
    sig_col, pred_col = st.columns([1.6, 1], gap="large")
    with sig_col:
        st.markdown("**Input signal**")
        st.pyplot(plot_signal(window), use_container_width=True)
    with pred_col:
        st.markdown("**Prediction**")
        st.markdown(
            f"<div style='font-size: 1.4rem; font-weight: 600; color: {TEXT}; "
            f"line-height: 1.15; margin: 0.25rem 0 0.5rem 0;'>{pred_name}</div>",
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{top_probs[0]:.1%}")
        m2.metric("Class id", str(pred_id))
        st.markdown(
            f'<span class="small-muted">Group · {gesture_group(pred_id)}</span>',
            unsafe_allow_html=True,
        )

        if true_id is not None:
            true_name = labels.get(true_id, f"class_{true_id}")
            if true_id == pred_id:
                st.success(f"Correct. True label: **{true_name}**.")
            elif true_id in top_idx.tolist():
                rank = top_idx.tolist().index(true_id) + 1
                st.warning(
                    f"Top-1 wrong. True label **{true_name}** ranked {rank} "
                    f"at {probs[true_id]:.1%}."
                )
            else:
                st.error(
                    f"Missed. True label **{true_name}** got "
                    f"{probs[true_id]:.1%}, not in top 5."
                )

    st.markdown("**Top 5 guesses**")
    correct_pos = None
    if true_id is not None and true_id in top_idx.tolist():
        correct_pos = top_idx.tolist().index(true_id)
    st.pyplot(plot_top5(top_probs, top_names, correct_pos), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="EMG gesture classifier",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="expanded",
    )
    inject_css()

    model, stats, labels, comparison = load_model()
    render_sidebar(comparison)

    if model is None:
        st.warning(
            f"No trained model found at `{CHECKPOINT.relative_to(ROOT)}`. "
            f"Train the CNN first and rerun the app."
        )
        st.stop()

    render_hero()
    render_demo(model, stats, labels)
    st.divider()
    render_intro_expanders(labels)


if __name__ == "__main__":
    main()
