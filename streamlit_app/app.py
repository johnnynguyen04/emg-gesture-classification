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


# Palette, taken from the AdventHealth logo spectrum: deep blue, sky blue,
# lime green, deep teal. The chart steps pass the CVD-separation and
# lightness checks against a white surface.
BLUE = "#005C95"
SKY = "#1BA8E1"
GREEN = "#82C342"
CHART_BLUE = "#1787C0"
CHART_GREEN = "#82C342"
INK = "#0C2D42"
BODY = "#3D5468"
MUTED = "#6E8296"
BORDER = "#DCE7EF"
BG = "#FCFDFE"
CARD = "#ffffff"
TINT = "#F1F7FA"

# Channels sweep the brand spectrum, deep blue through sky to green. The
# green endpoint is a darker step than the logo lime so thin lines stay
# readable on the light surface.
CHANNEL_CMAP = LinearSegmentedColormap.from_list(
    "channels", [BLUE, SKY, "#6CAB34"]
)

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Plus Jakarta Sans", "Segoe UI", "Arial", "sans-serif"],
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "regular",
    "axes.titlecolor": INK,
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
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html {{ scroll-behavior: smooth; }}
        .sec-num {{ scroll-margin-top: 90px; }}

        /* Static layered washes: blue from the top, a faint purple from the
           right edge. No animation — the page holds still. */
        .stApp {{
            background:
                radial-gradient(90rem 42rem at 12% -8%, rgba(0, 92, 149, 0.1), transparent 60%),
                radial-gradient(50rem 30rem at 85% -6%, rgba(27, 168, 225, 0.09), transparent 55%),
                radial-gradient(70rem 46rem at 108% 24%, rgba(130, 195, 66, 0.07), transparent 55%),
                {BG};
            background-attachment: fixed;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {BODY};
        }}

        .block-container {{ max-width: 1060px; padding-top: 2.5rem; }}
        section[data-testid="stSidebar"] {{ display: none; }}
        header[data-testid="stHeader"] {{ display: none; }}

        /* ------------------------------------------------------ type */
        .stApp h1 {{
            font-family: 'Fraunces', serif;
            font-weight: 550;
            font-size: 3.6rem !important;
            line-height: 1.04 !important;
            letter-spacing: -0.015em;
            color: {INK};
        }}
        .stApp h3 {{
            font-family: 'Fraunces', serif !important;
            font-weight: 500;
            font-size: 1.8rem !important;
            letter-spacing: -0.008em;
            color: {INK};
            margin-top: 0 !important;
            padding-bottom: 0.9rem;
            border-bottom: 2px solid;
            border-image: linear-gradient(90deg, {SKY}, {BORDER} 45%, transparent) 1;
        }}
        .small-muted {{ color: {MUTED}; font-size: 0.9rem; line-height: 1.6; }}

        /* --------------------------------------------------- metrics */
        [data-testid="stMetricValue"] {{
            font-family: 'Fraunces', serif !important;
            font-weight: 500;
            font-variant-numeric: tabular-nums;
            color: {BLUE};
            font-size: 2.7rem !important;
            line-height: 1.05 !important;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.68rem !important;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: {MUTED};
            font-weight: 600;
        }}
        [data-testid="stMetric"] {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.1rem 1.3rem 1rem 1.3rem;
            box-shadow: 0 18px 40px -32px rgba(0, 92, 149, 0.35);
            position: relative;
            overflow: hidden;
        }}
        [data-testid="stMetric"]::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, {BLUE}, {SKY} 55%, {GREEN});
        }}

        /* ------------------------------------------------ nav pill */
        .top-nav {{
            position: sticky;
            top: 0.9rem;
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 0.2rem;
            width: fit-content;
            margin: -1rem auto 2rem auto;
            padding: 0.45rem 0.6rem;
            background: rgba(255, 255, 255, 0.78);
            backdrop-filter: blur(16px) saturate(160%);
            -webkit-backdrop-filter: blur(16px) saturate(160%);
            border: 1px solid rgba(0, 92, 149, 0.10);
            border-radius: 999px;
            box-shadow: 0 12px 32px -18px rgba(0, 92, 149, 0.35);
        }}
        .top-nav .brand {{
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            font-family: 'Fraunces', serif;
            font-weight: 550;
            font-size: 0.92rem;
            color: {INK};
            padding: 0 0.9rem 0 0.7rem;
        }}
        .top-nav a {{
            font-size: 0.8rem;
            font-weight: 600;
            color: {MUTED};
            text-decoration: none;
            padding: 0.42rem 0.85rem;
            border-radius: 999px;
            transition: color 0.4s cubic-bezier(0.32, 0.72, 0, 1),
                        background 0.4s cubic-bezier(0.32, 0.72, 0, 1);
        }}
        .top-nav a:hover {{
            color: {BLUE};
            background: rgba(0, 92, 149, 0.07);
        }}

        /* ---------------------------------------------------- hero */
        .hero-block {{ padding: 1.5rem 0 0 0; }}
        .hero-grid {{
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 3.5rem;
            align-items: center;
        }}
        @media (max-width: 760px) {{
            .hero-grid {{ grid-template-columns: 1fr; gap: 2rem; }}
        }}
        .hero-eyebrow {{
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: {MUTED};
            margin-bottom: 1.2rem;
        }}
        .hero-sub {{
            font-size: 1.08rem;
            color: {BODY};
            line-height: 1.65;
            max-width: 50ch;
            margin: 1.1rem 0 2rem 0;
        }}
        .hero-stats {{ display: flex; gap: 2.75rem; flex-wrap: wrap; }}
        .hero-stat-value {{
            font-family: 'Fraunces', serif;
            font-size: 3.1rem;
            font-weight: 500;
            line-height: 1;
            background: linear-gradient(135deg, {BLUE} 20%, {SKY});
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            color: {BLUE};
        }}
        .hero-stat-value .unit {{ font-size: 0.55em; }}
        .hero-stat-label {{
            font-size: 0.66rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: {MUTED};
            font-weight: 600;
            margin-top: 0.5rem;
        }}
        .byline {{ color: {MUTED}; font-size: 0.88rem; margin: 1.6rem 0 1.1rem 0; }}

        .text-link {{
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 0.95rem;
            font-weight: 600;
            color: {BLUE} !important;
            text-decoration: none !important;
        }}
        .text-link .arrow {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.7rem;
            height: 1.7rem;
            border-radius: 999px;
            background: rgba(0, 92, 149, 0.08);
            font-size: 0.9rem;
            transition: transform 0.5s cubic-bezier(0.32, 0.72, 0, 1),
                        background 0.5s cubic-bezier(0.32, 0.72, 0, 1);
        }}
        .text-link:hover .arrow {{
            transform: translateX(3px);
            background: rgba(0, 92, 149, 0.14);
        }}

        /* Hero card: an outer tinted shell holding a white inner card, so it
           reads as one physical object rather than a floating rectangle. */
        .bezel {{
            background: linear-gradient(155deg, rgba(0, 92, 149, 0.08), rgba(27, 168, 225, 0.08), rgba(130, 195, 66, 0.07));
            border: 1px solid rgba(0, 92, 149, 0.12);
            border-radius: 24px;
            padding: 8px;
            box-shadow: 0 30px 60px -30px rgba(0, 92, 149, 0.35);
            animation: cardFloat 7s cubic-bezier(0.42, 0, 0.58, 1) infinite;
        }}
        @keyframes cardFloat {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-7px); }}
        }}
        .bezel-core {{
            background: {CARD};
            border-radius: 17px;
            padding: 1.5rem 1.6rem;
        }}
        .live-label {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: {BLUE};
            margin-bottom: 0.9rem;
        }}
        .live-dot {{
            width: 6px; height: 6px; border-radius: 50%;
            background: {GREEN};
            animation: dotBreathe 2.6s ease-in-out infinite;
        }}
        @keyframes dotBreathe {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(130, 195, 66, 0.5); }}
            55% {{ box-shadow: 0 0 0 6px rgba(130, 195, 66, 0); }}
        }}
        .pred-name {{
            font-family: 'Fraunces', serif;
            font-size: 1.5rem;
            font-weight: 550;
            color: {INK};
            margin-bottom: 0.8rem;
            line-height: 1.15;
        }}
        .pred-conf {{
            font-family: 'Fraunces', serif;
            font-size: 3.4rem;
            font-weight: 500;
            color: {BLUE};
            line-height: 1;
        }}
        .pred-conf .unit {{ font-size: 0.42em; margin-left: 0.1em; }}
        .pred-meta {{
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {MUTED};
            padding-top: 0.9rem;
            margin-top: 1.1rem;
            border-top: 1px solid {BORDER};
        }}
        .spec-rows {{ margin-top: 1rem; }}
        .spec-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 0.38rem 0.35rem;
            font-size: 0.8rem;
        }}
        .spec-row span:first-child {{ color: {MUTED}; }}
        .spec-row span:last-child {{
            color: {INK};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
        }}

        /* Flowy divider between the hero and the working sections. The svg
           is wider than its box and drifts slowly side to side. */
        .wave-sep {{ margin: 2.5rem -1rem 1.5rem -1rem; line-height: 0; overflow: hidden; }}
        .wave-sep svg {{
            width: 120%;
            margin-left: -10%;
            height: 54px;
            display: block;
            animation: waveDrift 16s cubic-bezier(0.42, 0, 0.58, 1) infinite alternate;
        }}
        @keyframes waveDrift {{
            from {{ transform: translateX(-3.5%); }}
            to {{ transform: translateX(3.5%); }}
        }}

        /* ------------------------------------------------- sections */
        .sec-num {{
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 500;
            color: {BLUE};
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin: 3.5rem 0 0.6rem 0;
        }}
        .section-note {{
            color: {MUTED};
            font-size: 0.94rem;
            line-height: 1.65;
            margin: 0.25rem 0 1.5rem 0;
            max-width: 64ch;
        }}
        .cm-label {{
            font-size: 0.72rem;
            font-weight: 700;
            color: {INK};
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.4rem;
        }}
        .prose {{
            font-size: 1.03rem;
            line-height: 1.75;
            color: {BODY};
            border-left: 3px solid;
            border-image: linear-gradient(180deg, {SKY}, {GREEN}) 1;
            padding-left: 1.3rem;
            margin: 0.5rem 0;
            max-width: 66ch;
        }}

        .method-card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
            height: 100%;
            box-shadow: 0 18px 40px -32px rgba(0, 92, 149, 0.35);
            transition: transform 0.5s cubic-bezier(0.32, 0.72, 0, 1),
                        box-shadow 0.5s cubic-bezier(0.32, 0.72, 0, 1);
        }}
        .method-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 26px 48px -30px rgba(0, 92, 149, 0.5);
        }}
        .method-title {{
            font-family: 'Fraunces', serif;
            font-size: 1.05rem;
            font-weight: 550;
            color: {INK};
            margin-bottom: 0.5rem;
        }}
        .method-num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.8rem;
            height: 1.8rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(0, 92, 149, 0.1), rgba(27, 168, 225, 0.12));
            color: {BLUE};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.05em;
            margin-bottom: 0.7rem;
        }}
        .method-body {{ font-size: 0.92rem; color: {BODY}; line-height: 1.65; }}

        /* ------------------------------------------------- widgets */
        .stButton button, .stLinkButton a, .stDownloadButton button {{
            border-radius: 999px !important;
            border: 1px solid {BORDER} !important;
            font-weight: 600 !important;
            transition: border-color 0.4s cubic-bezier(0.32, 0.72, 0, 1),
                        box-shadow 0.4s cubic-bezier(0.32, 0.72, 0, 1) !important;
        }}
        .stButton button:hover {{
            border-color: {BLUE} !important;
            box-shadow: 0 6px 18px -8px rgba(0, 92, 149, 0.4) !important;
        }}
        .stButton button:active {{ transform: scale(0.98); }}

        .stRadio > div {{ gap: 0.5rem !important; }}
        .stRadio label {{
            background: {CARD};
            border: 1px solid {BORDER};
            padding: 0.45rem 1.05rem !important;
            border-radius: 999px;
            cursor: pointer;
            transition: border-color 0.4s cubic-bezier(0.32, 0.72, 0, 1),
                        background 0.4s cubic-bezier(0.32, 0.72, 0, 1);
        }}
        .stRadio label:hover {{ border-color: {BLUE}; background: {TINT}; }}
        .stRadio label[data-checked="true"] {{ border-color: {BLUE}; background: {TINT}; }}

        .stSelectbox [data-baseweb="select"] > div {{
            border-radius: 12px !important;
            border-color: {BORDER} !important;
            background: {CARD} !important;
            transition: border-color 0.4s cubic-bezier(0.32, 0.72, 0, 1),
                        box-shadow 0.4s cubic-bezier(0.32, 0.72, 0, 1);
        }}
        .stSelectbox [data-baseweb="select"] > div:hover {{ border-color: {BLUE} !important; }}
        .stSelectbox [data-baseweb="select"] > div:focus-within {{
            border-color: {BLUE} !important;
            box-shadow: 0 0 0 3px rgba(0, 92, 149, 0.12);
        }}

        [data-testid="stFileUploader"] section {{
            background: {TINT};
            border: 1.5px dashed rgba(0, 92, 149, 0.25);
            border-radius: 16px;
        }}

        [data-testid="stExpander"] {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            margin-bottom: 0.6rem;
            box-shadow: 0 14px 32px -28px rgba(0, 92, 149, 0.3);
            transition: border-color 0.4s cubic-bezier(0.32, 0.72, 0, 1);
        }}
        [data-testid="stExpander"]:hover {{ border-color: rgba(0, 92, 149, 0.3); }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details > summary {{
            font-weight: 600 !important;
            color: {INK} !important;
            padding: 0.4rem 0 !important;
        }}

        [data-testid="stAlert"] {{
            border-radius: 14px;
            border: 1px solid {BORDER};
            padding: 0.85rem 1rem !important;
        }}
        hr {{ border-color: {BORDER}; }}

        /* -------------------------------------------------- footer */
        .footer-card {{
            background: linear-gradient(160deg, rgba(0, 92, 149, 0.05), rgba(27, 168, 225, 0.05), rgba(130, 195, 66, 0.05));
            border: 1px solid rgba(0, 92, 149, 0.1);
            border-radius: 24px;
            padding: 2.4rem 2.2rem 1.6rem 2.2rem;
            margin-top: 4rem;
        }}
        .footer-meta {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2.5rem;
            margin-bottom: 1.75rem;
        }}
        @media (max-width: 640px) {{
            .footer-meta {{ grid-template-columns: 1fr; gap: 1.5rem; }}
        }}
        .meta-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: {BLUE};
            margin-bottom: 0.85rem;
            font-weight: 500;
        }}
        .meta-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(0, 92, 149, 0.08);
            font-size: 0.9rem;
        }}
        .meta-row:last-child {{ border-bottom: none; }}
        .meta-key {{ color: {MUTED}; }}
        .meta-val {{
            color: {INK};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            text-align: right;
        }}
        .footer-bar {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            padding-top: 1.25rem;
            border-top: 1px solid rgba(0, 92, 149, 0.1);
        }}
        @media (max-width: 640px) {{
            .footer-bar {{ flex-direction: column; align-items: flex-start; }}
        }}
        .footer-name {{
            font-family: 'Fraunces', serif;
            font-size: 1.1rem;
            font-weight: 550;
            color: {INK};
            margin-bottom: 0.2rem;
        }}
        .footer-bio {{ color: {MUTED}; font-size: 0.84rem; }}
        .footer-links {{ display: flex; gap: 0.5rem; }}
        .footer-links a {{
            color: {BLUE};
            text-decoration: none;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.4rem 0.9rem;
            border: 1px solid rgba(0, 92, 149, 0.18);
            border-radius: 999px;
            transition: background 0.4s cubic-bezier(0.32, 0.72, 0, 1),
                        border-color 0.4s cubic-bezier(0.32, 0.72, 0, 1);
        }}
        .footer-links a:hover {{
            background: rgba(0, 92, 149, 0.07);
            border-color: rgba(0, 92, 149, 0.4);
        }}
        .copyright {{
            color: {MUTED};
            font-size: 0.72rem;
            margin-top: 1.25rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(0, 92, 149, 0.08);
        }}

        /* -------------------------------------------------- motion */
        /* One system. Entry rise for the hero and nav on load; a scroll-tied
           rise for everything below the fold. fill-mode backwards keeps
           elements visible if an animation never fires. */
        @keyframes riseIn {{
            from {{ opacity: 0; transform: translateY(18px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .hero-block {{ animation: riseIn 0.8s cubic-bezier(0.32, 0.72, 0, 1) backwards; }}
        .top-nav {{ animation: riseIn 0.6s cubic-bezier(0.32, 0.72, 0, 1) backwards; }}

        @keyframes smoothFade {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .pred-reveal {{ animation: smoothFade 0.5s cubic-bezier(0.32, 0.72, 0, 1); }}

        /* CSS view() timeline (Chrome/Edge 115+, Safari 26+). Older browsers
           ignore animation-timeline and play the rise once on load instead. */
        @keyframes scrollRise {{
            from {{ opacity: 0; transform: translateY(36px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        [data-testid="stPyplotChart"],
        [data-testid="stImage"],
        [data-testid="stExpander"],
        .method-card,
        .prose,
        .footer-card {{
            animation: scrollRise linear both;
            animation-timeline: view();
            animation-range: entry 0% cover 26%;
        }}

        @media (prefers-reduced-motion: reduce) {{
            .bezel, .live-dot, .wave-sep svg,
            .hero-block, .top-nav, .pred-reveal {{ animation: none !important; }}
        }}

        /* Small screens: tighter type, brand-only nav pill. */
        @media (max-width: 640px) {{
            .stApp h1 {{ font-size: 2.4rem !important; }}
            .stApp h3 {{ font-size: 1.5rem !important; }}
            .top-nav {{ padding: 0.4rem 0.55rem; }}
            .top-nav a {{ display: none; }}
            .hero-stat-value {{ font-size: 2.5rem; }}
            .hero-stats {{ gap: 1.8rem; }}
            .pred-conf {{ font-size: 2.7rem; }}
            [data-testid="stMetricValue"] {{ font-size: 2.1rem !important; }}
            .block-container {{ padding-top: 1.5rem; }}
            .footer-card {{ padding: 1.6rem 1.3rem 1.2rem 1.3rem; }}
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
        colors = [CHART_BLUE] * len(top_probs)
        if correct_idx is not None:
            colors[correct_idx] = CHART_GREEN
        ax.barh(range(len(top_probs)), top_probs[::-1], color=colors[::-1], height=0.62)
        ax.set_yticks(range(len(top_probs)))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability")
        for i, p in enumerate(top_probs[::-1]):
            ax.text(p + 0.012, i, f"{p:.1%}", va="center", fontsize=9, color=INK,
                    family="monospace")
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ax.tick_params(axis="y", length=0)
        if correct_idx is not None:
            ax.text(1.0, -0.9, "true label", color=CHART_GREEN, fontsize=8.5,
                    ha="right", va="center", family="monospace")
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


def render_nav() -> None:
    st.markdown(
        """
        <div class="top-nav">
            <span class="brand">
                <svg width="20" height="14" viewBox="0 0 20 14" aria-hidden="true">
                    <defs><linearGradient id="wv" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0" stop-color="#005C95"/>
                        <stop offset="0.55" stop-color="#1BA8E1"/>
                        <stop offset="1" stop-color="#82C342"/>
                    </linearGradient></defs>
                    <path d="M1 7 h3.5 l1.8 -4.5 2.8 9 2 -6.5 1.7 2 h6.2"
                          fill="none" stroke="url(#wv)" stroke-width="1.8"
                          stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                EMG Gesture Classifier
            </span>
            <a href="#classify">Classify</a>
            <a href="#compare">Compare</a>
            <a href="#perclass">Classes</a>
            <a href="#method">Method</a>
            <a href="#why">Why</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(model, stats, labels: dict, comparison: dict) -> None:
    cnn = comparison.get("cnn_1d", {})

    # Run the model on a real held-out window so the hero card shows an
    # actual prediction, not a mockup.
    card_html = ""
    sample_path = SAMPLES_DIR / "sample_class05.npy"
    if model is not None and sample_path.exists():
        window = np.load(sample_path).astype(np.float32)
        probs = predict(model, window, stats)
        pred_id = int(probs.argmax())
        pred_name = labels.get(pred_id, f"class_{pred_id}")
        conf = probs[pred_id]
        true_id = true_class_from_filename(sample_path.name)
        outcome = "correct" if pred_id == true_id else f"true class {true_id}"
        card_html = f"""
            <div class="bezel">
                <div class="bezel-core">
                    <div class="live-label"><span class="live-dot"></span>Model output · test window</div>
                    <div class="pred-name">{pred_name}</div>
                    <div class="pred-conf">{conf:.1%}</div>
                    <div class="pred-meta"><span>{gesture_group(pred_id)}</span><span>{outcome}</span></div>
                    <div class="spec-rows">
                        <div class="spec-row"><span>Architecture</span><span>1D CNN · 60,309 params</span></div>
                        <div class="spec-row"><span>Test accuracy</span><span>{cnn.get('accuracy', 0):.1%}</span></div>
                    </div>
                </div>
            </div>
        """

    st.markdown(
        f"""
        <div class="hero-block hero-grid">
            <div>
                <span class="hero-eyebrow">Surface EMG · NinaPro DB1</span>
                <h1 style="margin: 0;">Reading gestures<br>from muscle signals</h1>
                <div class="hero-sub">
                    1.2 million surface-EMG windows from 27 subjects, classifying
                    53 hand gestures with a Random Forest baseline and a 1D CNN.
                </div>
                <div class="hero-stats">
                    <div>
                        <div class="hero-stat-value">1.2<span class="unit">M</span></div>
                        <div class="hero-stat-label">Windows</div>
                    </div>
                    <div>
                        <div class="hero-stat-value">27</div>
                        <div class="hero-stat-label">Subjects</div>
                    </div>
                    <div>
                        <div class="hero-stat-value">53</div>
                        <div class="hero-stat-label">Gestures</div>
                    </div>
                </div>
                <div class="byline">by Johnny Nguyen · UCF Data Science</div>
                <a href="{GITHUB_URL}" target="_blank" class="text-link">
                    View source on GitHub <span class="arrow">→</span>
                </a>
            </div>
            <div>{card_html}</div>
        </div>
        <div class="wave-sep">
            <svg viewBox="0 0 1440 54" preserveAspectRatio="none">
                <path d="M0,30 C240,54 480,4 720,18 C960,32 1200,50 1440,24 L1440,54 L0,54 Z"
                      fill="rgba(27, 168, 225, 0.12)"/>
                <path d="M0,40 C280,58 520,14 760,26 C1000,38 1240,54 1440,34 L1440,54 L0,54 Z"
                      fill="rgba(130, 195, 66, 0.1)"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo(model, stats, labels) -> None:
    st.markdown('<span class="sec-num" id="classify">§ 01 — Try the model</span><h3>Live classification</h3>', unsafe_allow_html=True)
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
        st.markdown(f'<div style="color:{MUTED}; font-size: 0.72rem; '
                    f'text-transform: uppercase; letter-spacing: 0.1em; '
                    f'font-weight: 600;">Signal trace</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_signal(window), use_container_width=True)
    with pred_col:
        st.markdown(
            f'<div class="pred-reveal">'
            f'<div style="color:{MUTED}; font-size: 0.72rem; '
            f'text-transform: uppercase; letter-spacing: 0.1em; '
            f'font-weight: 600; margin-bottom: 0.4rem;">Prediction</div>'
            f'<div class="pred-name">{pred_name}</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; '
            f'font-size: 1.1rem; color: {BLUE}; margin-bottom: 0.35rem;">'
            f'{top_probs[0]:.1%} confidence</div>'
            f'<div class="small-muted">group · {gesture_group(pred_id)}<br>'
            f'class id · {pred_id}</div>'
            f'</div>',
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
    st.markdown('<span class="sec-num">§ 02 — Ranked confidence</span><h3>Top 5 predictions</h3>', unsafe_allow_html=True)
    correct_pos = None
    if true_id is not None and true_id in top_idx.tolist():
        correct_pos = top_idx.tolist().index(true_id)
    st.pyplot(plot_top5(top_probs, top_names, correct_pos), use_container_width=True)


def plot_per_class(rows: list[dict]) -> plt.Figure:
    """Horizontal F1 bars for a small set of gestures."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        names = [r["name"] if len(r["name"]) <= 28 else r["name"][:26] + "…"
                 for r in rows[::-1]]
        f1s = [r["f1"] for r in rows[::-1]]
        ax.barh(range(len(rows)), f1s, color=CHART_BLUE, height=0.62)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1")
        for i, f in enumerate(f1s):
            ax.text(f + 0.012, i, f"{f:.2f}", va="center", fontsize=8,
                    color=INK, family="monospace")
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ax.tick_params(axis="y", length=0)
        fig.tight_layout()
    return fig


def render_model_comparison(comparison: dict) -> None:
    st.markdown('<span class="sec-num" id="compare">§ 03 — Head to head</span><h3>Model comparison</h3>', unsafe_allow_html=True)
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

    st.markdown('<span class="sec-num" id="perclass">§ 04 — Where the model breaks</span><h3>Per-class performance</h3>', unsafe_allow_html=True)
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
        st.pyplot(plot_per_class(bottom), use_container_width=True)
    with c2:
        st.markdown('<div class="cm-label">Easiest 10 classes</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_per_class(top), use_container_width=True)


def render_methodology() -> None:
    st.markdown('<span class="sec-num" id="method">§ 05 — How it works</span><h3>Methodology</h3>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">How this was built, four steps.</div>',
        unsafe_allow_html=True,
    )

    cards = [
        ("Preprocess",
         "Load 27 subjects from NinaPro DB1 (10 channels, 100 Hz, Otto Bock "
         "electrodes that hardware-rectify the signal). Slide 200 ms windows "
         "with 100 ms overlap. Drop any window crossing a gesture boundary so "
         "the labels stay clean."),
        ("Features (RF only)",
         "Hudgins time-domain set per channel: mean absolute value, zero "
         "crossings, slope sign changes, waveform length, RMS, variance. "
         "Six features × ten channels = sixty per window."),
        ("Model",
         "1D CNN: three Conv1D blocks (32 → 64 → 128 channels, kernel 5) with "
         "batchnorm and ReLU, max-pool between them, adaptive average pool, "
         "dropout 0.3, linear head to 53 classes. About 60k parameters total."),
        ("Train",
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
                f'<div class="method-num">{i + 1:02d}</div>'
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
    st.markdown('<span class="sec-num" id="why">§ 06 — Context</span><h3>Why I built this</h3>', unsafe_allow_html=True)
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


def render_footer(comparison: dict) -> None:
    cnn = comparison.get("cnn_1d", {})
    rf = comparison.get("random_forest", {})
    rows = [
        ("dataset", [
            ("Source", "NinaPro DB1"),
            ("Records", "1,201,994 windows"),
            ("Subjects", "27"),
            ("Gestures", "53"),
            ("Channels", "10"),
            ("Sampling", "100 Hz · Otto Bock"),
        ]),
        ("model", [
            ("Architecture", "1D CNN (Conv1D ×3)"),
            ("Parameters", "60,309"),
            ("Test accuracy", f"{cnn.get('accuracy', 0):.1%}"),
            ("Macro F1", f"{cnn.get('macro_f1', 0):.3f}"),
            ("Baseline RF", f"{rf.get('accuracy', 0):.1%}"),
            ("Train epochs", "30"),
        ]),
    ]

    cols_html = ""
    for title, items in rows:
        rows_html = "".join(
            f'<div class="meta-row"><span class="meta-key">{k}</span>'
            f'<span class="meta-val">{v}</span></div>'
            for k, v in items
        )
        cols_html += (
            f'<div class="meta-col">'
            f'<div class="meta-title">{title}</div>{rows_html}</div>'
        )

    st.markdown(
        f"""
        <div class="footer-card">
            <div class="footer-meta">{cols_html}</div>
            <div class="footer-bar">
                <div>
                    <div class="footer-name">EMG Hand Gesture Classifier</div>
                    <div class="footer-bio">By Johnny Nguyen · UCF Data Science · Independent study</div>
                </div>
                <div class="footer-links">
                    <a href="{GITHUB_URL}" target="_blank">GitHub</a>
                    <a href="https://www.linkedin.com/in/johnnynguyen04/" target="_blank">LinkedIn</a>
                    <a href="mailto:johnny060904@gmail.com">Email</a>
                </div>
            </div>
            <div class="copyright">© 2026 Johnny Nguyen. NinaPro DB1 dataset used under its terms (Atzori et al., 2014).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="EMG Hand Gesture Classifier",
        page_icon=str(ROOT / "streamlit_app" / "assets" / "favicon.png"),
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

    render_nav()
    render_hero(model, stats, labels, comparison)
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
    render_footer(comparison)


if __name__ == "__main__":
    main()
