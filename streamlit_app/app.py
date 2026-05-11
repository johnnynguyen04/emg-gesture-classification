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
RF_METRICS = ROOT / "results" / "metrics" / "rf_metrics.json"
CNN_METRICS = ROOT / "results" / "metrics" / "cnn_metrics.json"
RF_CONFUSION = ROOT / "results" / "figures" / "rf_confusion.png"
CNN_CONFUSION = ROOT / "results" / "figures" / "cnn_confusion.png"
SAMPLES_DIR = ROOT / "streamlit_app" / "samples"
GITHUB_URL = "https://github.com/johnnynguyen04/emg-gesture-classification"

PRIMARY = "#0e7fb8"
PRIMARY_DARK = "#0a6da0"
ACCENT = "#5fa238"
TEXT = "#0f172a"
TEXT_BODY = "#334155"
MUTED = "#64748b"
BORDER = "#e2e8f0"
BG = "#ffffff"
CARD = "#ffffff"
CARD_TINT = "#f8fafc"
CHANNEL_CMAP = LinearSegmentedColormap.from_list(
    "channels", ["#93c5e8", "#0e7fb8", "#0a4a70"]
)

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter Tight", "Inter", "Helvetica", "Arial", "sans-serif"],
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
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        .stApp {{
            background: {BG};
            font-family: 'Inter Tight', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {TEXT_BODY};
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
            font-family: 'Fraunces', Georgia, serif;
            font-weight: 500;
            letter-spacing: -0.01em;
            color: {TEXT};
        }}
        .stApp h1 {{ color: {PRIMARY}; font-weight: 600; }}
        [data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 500;
            color: {PRIMARY};
            font-size: 1.7rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.7rem !important;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: {MUTED};
            font-weight: 500;
        }}
        [data-testid="stMetric"] {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(14, 127, 184, 0.04);
        }}
        .stButton button, .stLinkButton a, .stDownloadButton button {{
            border-radius: 8px !important;
            border: 1px solid {BORDER} !important;
        }}
        hr {{ border-color: {BORDER}; }}
        .small-muted {{ color: {MUTED}; font-size: 0.9rem; line-height: 1.6; }}
        .byline {{
            color: {MUTED};
            font-size: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.02em;
        }}
        .prose {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.05rem;
            line-height: 1.7;
            color: {TEXT_BODY};
            border-left: 2px solid {ACCENT};
            padding-left: 1.25rem;
            margin: 0.5rem 0;
        }}
        .footer-card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1.75rem;
            text-align: center;
            margin-top: 2rem;
        }}
        .avatar {{
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_DARK});
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-family: 'Fraunces', serif;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }}
        .footer-name {{
            font-family: 'Fraunces', serif;
            font-size: 1.15rem;
            color: {TEXT};
            margin-bottom: 0.25rem;
            font-weight: 500;
        }}
        .footer-bio {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 1rem; }}
        .footer-links a {{
            color: {PRIMARY};
            text-decoration: none;
            font-size: 0.9rem;
            margin: 0 0.6rem;
            font-weight: 500;
        }}
        .footer-links a:hover {{ text-decoration: underline; }}
        .copyright {{ color: {MUTED}; font-size: 0.78rem; margin-top: 1rem; }}
        section[data-testid="stSidebar"] {{ display: none; }}
        .block-container {{ max-width: 760px; padding-top: 2rem; }}

        .method-card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            height: 100%;
            transition: box-shadow 0.18s ease, transform 0.18s ease;
        }}
        .method-card:hover {{
            box-shadow: 0 4px 14px rgba(14, 127, 184, 0.08);
            transform: translateY(-1px);
        }}
        .method-title {{
            font-family: 'Fraunces', serif;
            font-size: 1.05rem;
            color: {PRIMARY};
            margin-bottom: 0.5rem;
            font-weight: 500;
            letter-spacing: -0.01em;
        }}
        .method-body {{
            font-size: 0.92rem;
            color: {TEXT_BODY};
            line-height: 1.6;
        }}
        .section-note {{
            color: {MUTED};
            font-size: 0.9rem;
            line-height: 1.55;
            margin: 0.25rem 0 1.25rem 0;
            max-width: 60ch;
        }}
        .cm-label {{
            font-family: 'Fraunces', serif;
            color: {TEXT};
            font-size: 0.95rem;
            text-align: center;
            margin-bottom: 0.25rem;
        }}

        /* Streamlit alerts */
        [data-testid="stAlert"] {{ border-radius: 10px; }}

        /* Entry animation: staggered fade-up on first paint */
        @keyframes fadeUp {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .stApp .element-container {{
            animation: fadeUp 0.45s ease-out both;
        }}
        .stApp .element-container:nth-child(1) {{ animation-delay: 0ms; }}
        .stApp .element-container:nth-child(2) {{ animation-delay: 60ms; }}
        .stApp .element-container:nth-child(3) {{ animation-delay: 120ms; }}
        .stApp .element-container:nth-child(4) {{ animation-delay: 180ms; }}
        .stApp .element-container:nth-child(5) {{ animation-delay: 240ms; }}
        .stApp .element-container:nth-child(6) {{ animation-delay: 300ms; }}
        .stApp .element-container:nth-child(7) {{ animation-delay: 360ms; }}
        .stApp .element-container:nth-child(8) {{ animation-delay: 420ms; }}
        .stApp .element-container:nth-child(9) {{ animation-delay: 480ms; }}
        .stApp .element-container:nth-child(10) {{ animation-delay: 520ms; }}

        /* Card hover lift */
        [data-testid="stMetric"] {{
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        [data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(14, 127, 184, 0.10);
        }}
        .footer-card {{
            transition: box-shadow 0.18s ease;
        }}
        .footer-card:hover {{
            box-shadow: 0 6px 18px rgba(14, 127, 184, 0.08);
        }}

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
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        t = np.arange(n_t) * 1000 / 100
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
        fig, ax = plt.subplots(figsize=(7.0, 2.8))
        colors = [PRIMARY] * len(top_probs)
        if correct_idx is not None:
            colors[correct_idx] = ACCENT
        ax.barh(range(len(top_probs)), top_probs[::-1], color=colors[::-1], height=0.62)
        ax.set_yticks(range(len(top_probs)))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability")
        for i, p in enumerate(top_probs[::-1]):
            ax.text(p + 0.012, i, f"{p:.1%}", va="center", fontsize=9, color=TEXT,
                    family="JetBrains Mono")
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ax.tick_params(axis="y", length=0)
        if correct_idx is not None:
            ax.text(1.0, -0.9, "true label", color=ACCENT, fontsize=8.5,
                    ha="right", va="center", family="JetBrains Mono")
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


def render_hero() -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 1.75rem;">
            <h1 style="font-size: 2.5rem; margin: 0 0 0.5rem 0;">EMG hand gesture classifier</h1>
            <div class="byline">by Johnny Nguyen · UCF Data Science</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats(comparison: dict) -> None:
    cnn = comparison.get("cnn_1d", {})
    rf = comparison.get("random_forest", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Subjects", "27")
    c2.metric("Gestures", "53")
    c3.metric("Windows trained", "350k")
    st.markdown("&nbsp;", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    c4.metric("RF test acc",
              f"{rf.get('accuracy', 0):.1%}" if rf else "—")
    c5.metric("CNN test acc",
              f"{cnn.get('accuracy', 0):.1%}" if cnn else "—")
    c6.metric("Delta", f"{(cnn.get('accuracy', 0) - rf.get('accuracy', 0)):+.1%}"
              if cnn and rf else "—")


def render_demo(model, stats, labels) -> None:
    st.markdown("### Live classification")
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

    sig_col, pred_col = st.columns([1.6, 1], gap="large")
    with sig_col:
        st.markdown(f'<div style="color:{MUTED}; font-size: 0.75rem; '
                    f'text-transform: uppercase; letter-spacing: 0.08em; '
                    f'font-weight: 500;">Signal trace</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_signal(window), use_container_width=True)
    with pred_col:
        st.markdown(f'<div style="color:{MUTED}; font-size: 0.75rem; '
                    f'text-transform: uppercase; letter-spacing: 0.08em; '
                    f'font-weight: 500; margin-bottom: 0.4rem;">Prediction</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family: \'Fraunces\', serif; font-size: 1.5rem; '
            f'font-weight: 500; color: {TEXT}; line-height: 1.15; '
            f'margin-bottom: 0.5rem;">{pred_name}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-family: \'JetBrains Mono\', monospace; '
            f'font-size: 1.1rem; color: {PRIMARY}; margin-bottom: 0.25rem;">'
            f'{top_probs[0]:.1%} confidence</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="small-muted">group · {gesture_group(pred_id)}<br>'
            f'class id · {pred_id}</div>',
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

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("### Top 5 predictions")
    correct_pos = None
    if true_id is not None and true_id in top_idx.tolist():
        correct_pos = top_idx.tolist().index(true_id)
    st.pyplot(plot_top5(top_probs, top_names, correct_pos), use_container_width=True)


def plot_per_class(rows: list[dict], color: str) -> plt.Figure:
    """Horizontal F1 bars for a small set of gestures."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        names = [r["name"] if len(r["name"]) <= 28 else r["name"][:26] + "…"
                 for r in rows[::-1]]
        f1s = [r["f1"] for r in rows[::-1]]
        ax.barh(range(len(rows)), f1s, color=color, height=0.62)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1")
        for i, f in enumerate(f1s):
            ax.text(f + 0.012, i, f"{f:.2f}", va="center", fontsize=8,
                    color=TEXT, family="JetBrains Mono")
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ax.tick_params(axis="y", length=0)
        fig.tight_layout()
    return fig


def render_model_comparison(comparison: dict) -> None:
    st.markdown("### Model comparison")
    st.markdown(
        '<div class="section-note">Random Forest on Hudgins time-domain '
        "features against a 1D CNN trained end to end on the same windows. "
        "Identical train/val/test split, no augmentation on either side, so "
        "the gap between them is honest."
        "</div>",
        unsafe_allow_html=True,
    )

    rf = comparison.get("random_forest", {})
    cnn = comparison.get("cnn_1d", {})
    delta = comparison.get("delta_accuracy", 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest", f"{rf.get('accuracy', 0):.1%}",
              help="test accuracy")
    c2.metric("1D CNN", f"{cnn.get('accuracy', 0):.1%}",
              delta=f"{delta:+.1%}", delta_color="inverse",
              help="test accuracy")
    c3.metric("Macro F1 (CNN)", f"{cnn.get('macro_f1', 0):.3f}")

    st.markdown("&nbsp;", unsafe_allow_html=True)
    if RF_CONFUSION.exists() and CNN_CONFUSION.exists():
        cm1, cm2 = st.columns(2)
        with cm1:
            st.markdown('<div class="cm-label">Random Forest</div>',
                        unsafe_allow_html=True)
            st.image(str(RF_CONFUSION), use_container_width=True)
        with cm2:
            st.markdown('<div class="cm-label">1D CNN</div>',
                        unsafe_allow_html=True)
            st.image(str(CNN_CONFUSION), use_container_width=True)
        st.markdown(
            '<div class="section-note" style="margin-top: 0.5rem;">'
            "Both matrices are row-normalized. The CNN's off-diagonal mass "
            "concentrates in the functional grasp block (classes 30–52), "
            "which is also where the per-class F1 drops below 0.20."
            "</div>",
            unsafe_allow_html=True,
        )


def render_per_class(labels: dict) -> None:
    if not CNN_METRICS.exists():
        return
    metrics = json.loads(CNN_METRICS.read_text())
    rows = []
    for k, v in metrics.items():
        if not k.isdigit():
            continue
        cid = int(k)
        if v.get("support", 0) == 0:
            continue
        rows.append({
            "class_id": cid,
            "name": labels.get(cid, f"class_{cid}"),
            "f1": float(v.get("f1-score", 0.0)),
            "support": int(v.get("support", 0)),
        })
    if not rows:
        return
    rows.sort(key=lambda r: r["f1"])
    bottom = rows[:10]
    top = rows[-10:][::-1]

    st.markdown("### Per-class performance")
    st.markdown(
        '<div class="section-note">CNN F1 score per gesture on the held-out '
        "test set. Hardest classes cluster in functional grasps where "
        "multi-muscle coordination patterns are subtle. Easiest are the "
        "basic finger movements with clean single-channel activations."
        "</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="cm-label">Hardest 10 classes</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_per_class(bottom, PRIMARY), use_container_width=True)
    with c2:
        st.markdown('<div class="cm-label">Easiest 10 classes</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_per_class(top, ACCENT), use_container_width=True)


def render_methodology() -> None:
    st.markdown("### Methodology")
    st.markdown(
        '<div class="section-note">How this was built, four steps.</div>',
        unsafe_allow_html=True,
    )

    cards = [
        ("1 · Preprocess",
         "Load 27 subjects from NinaPro DB1 (10 channels, 100 Hz, Otto Bock "
         "electrodes that hardware-rectify the signal). Slide 200 ms windows "
         "with 100 ms overlap. Drop any window crossing a gesture boundary so "
         "the labels stay clean."),
        ("2 · Features (RF only)",
         "Hudgins time-domain set per channel: mean absolute value, zero "
         "crossings, slope sign changes, waveform length, RMS, variance. "
         "Six features × ten channels = sixty per window."),
        ("3 · Model",
         "1D CNN: three Conv1D blocks (32 → 64 → 128 channels, kernel 5) with "
         "batchnorm and ReLU, max-pool between them, adaptive average pool, "
         "dropout 0.3, linear head to 53 classes. About 60k parameters total."),
        ("4 · Train",
         "Split by repetition (1–7 train, 8 val, 9–10 test) so the model "
         "never sees windows from a repetition it was trained on. Adam at "
         "lr 2e-3, batch 512, cross-entropy loss, early stopping on val "
         "accuracy. 30 epoch cap."),
    ]

    cols = st.columns(2)
    for i, (title, body) in enumerate(cards):
        with cols[i % 2]:
            st.markdown(
                f'<div class="method-card">'
                f'<div class="method-title">{title}</div>'
                f'<div class="method-body">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_expanders(labels: dict) -> None:
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
            "- **Signal trace**: ten lines, one per forearm electrode, stacked "
            "vertically. NinaPro DB1 uses Otto Bock electrodes that rectify and "
            "smooth the signal in hardware, so it looks calmer than raw EMG.\n"
            "- **Prediction**: the gesture the model is betting on, with its "
            "confidence and which group it belongs to.\n"
            "- **Top 5**: the model's five most likely guesses with probability. "
            "Low max probability means the model knows it is unsure. The green "
            "bar marks the true label when it is in the top 5."
        )


def render_why() -> None:
    st.markdown("### Why I built this")
    st.markdown(
        '<div class="prose">'
        "I wanted to take a biosignal project end to end: load raw EMG, "
        "preprocess it properly, train two classifiers worth comparing, and "
        "write up where each one wins. EMG-driven gesture recognition is the "
        "same family of problem a clinical EMG system has to solve, "
        "and that overlap is what drew me to it."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="prose" style="margin-top: 1rem;">'
        "I picked NinaPro DB1 over the newer DB2 or DB5 because DB1 has 53 "
        "gesture classes and uses a pre-rectified EMG envelope, the kind of "
        "signal a real clinical electrode setup tends to produce. The honest "
        "result of this run: my small CNN underperformed the Random Forest "
        "baseline by 18 points. The README on GitHub walks through why and "
        "what I would do differently next time."
        "</div>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer-card">
            <div class="avatar">JN</div>
            <div class="footer-name">Johnny Nguyen</div>
            <div class="footer-bio">UCF Data Science</div>
            <div class="footer-links">
                <a href="{GITHUB_URL}" target="_blank">GitHub</a>
                <a href="https://www.linkedin.com/in/johnnynguyen04/" target="_blank">LinkedIn</a>
                <a href="mailto:johnny060904@gmail.com">Email</a>
            </div>
            <div class="copyright">© 2026 Johnny Nguyen · EMG gesture classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="EMG hand gesture classifier",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    model, stats, labels, comparison = load_model()
    if model is None:
        st.warning(
            f"No trained model found at `{CHECKPOINT.relative_to(ROOT)}`. "
            f"Train the CNN first and rerun the app."
        )
        st.stop()

    render_hero()
    render_stats(comparison)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    render_demo(model, stats, labels)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    render_model_comparison(comparison)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    render_per_class(labels)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    render_methodology()
    st.markdown("&nbsp;", unsafe_allow_html=True)
    render_expanders(labels)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    render_why()
    render_footer()


if __name__ == "__main__":
    main()
