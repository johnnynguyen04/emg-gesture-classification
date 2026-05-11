"""Streamlit demo: predict a hand gesture from a 200 ms × 10ch sEMG window."""
from __future__ import annotations

import base64
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
ASSETS_DIR = ROOT / "streamlit_app" / "assets"
GITHUB_URL = "https://github.com/johnnynguyen04/emg-gesture-classification"


@st.cache_resource
def portrait_data_url() -> str | None:
    if not ASSETS_DIR.exists():
        return None
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        p = ASSETS_DIR / f"portrait{ext}"
        if p.exists():
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            data = base64.b64encode(p.read_bytes()).decode()
            return f"data:{mime};base64,{data}"
    return None



PRIMARY = "#1FA5DB"
PRIMARY_DARK = "#1B365D"
ACCENT = "#7BC043"
ACCENT_DARK = "#2A5D43"
TEXT = "#0F1F35"
TEXT_BODY = "#3A4B61"
MUTED = "#6B7A8F"
BORDER = "#dde4ed"
BG = "#ffffff"
CARD = "#ffffff"
CARD_TINT = "#f4f8fc"
CHANNEL_CMAP = LinearSegmentedColormap.from_list(
    "channels", ["#a5d6ed", "#1FA5DB", "#0F4B70"]
)

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Manrope", "Inter", "Helvetica", "Arial", "sans-serif"],
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
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400;1,9..144,500&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        .stApp {{
            background:
                radial-gradient(120% 95% at 50% 0%,
                    #ffffff 26%,
                    rgba(31, 165, 219, 0.09) 48%,
                    rgba(123, 192, 67, 0.06) 64%,
                    rgba(31, 165, 219, 0.14) 80%,
                    rgba(27, 54, 93, 0.10) 100%
                );
            background-attachment: fixed;
            background-size: 100% 100%;
            font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {TEXT_BODY};
            animation: bgBreathe 10s ease-in-out infinite;
        }}
        @keyframes bgBreathe {{
            0% {{ background-size: 100% 100%; background-position: 50% 0%; }}
            50% {{ background-size: 114% 114%; background-position: 44% 6%; }}
            100% {{ background-size: 100% 100%; background-position: 50% 0%; }}
        }}
        .stApp h1, .stApp h2 {{
            font-family: 'Manrope', sans-serif;
            font-weight: 700;
            letter-spacing: -0.028em;
            color: {PRIMARY_DARK};
        }}
        .stApp h1 {{
            font-weight: 800;
            font-size: 2.9rem !important;
            line-height: 1.05 !important;
            color: {PRIMARY_DARK};
        }}
        .stApp h3 {{
            font-family: 'Manrope', sans-serif !important;
            font-weight: 700;
            font-size: 1.65rem !important;
            color: {PRIMARY_DARK};
            letter-spacing: -0.02em;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'Manrope', sans-serif !important;
            font-weight: 700;
            color: {PRIMARY_DARK};
            font-size: 3.2rem !important;
            line-height: 1 !important;
            letter-spacing: -0.04em;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.7rem !important;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {MUTED};
            font-weight: 500;
            margin-top: 0.5rem;
        }}
        [data-testid="stMetric"] {{
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0.5rem 0;
            box-shadow: none;
        }}
        .stButton button, .stLinkButton a, .stDownloadButton button {{
            border-radius: 8px !important;
            border: 1px solid {BORDER} !important;
            font-weight: 600 !important;
        }}
        hr {{ border-color: {BORDER}; }}
        .small-muted {{ color: {MUTED}; font-size: 0.9rem; line-height: 1.6; }}
        .byline {{
            color: {MUTED};
            font-size: 0.9rem;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.02em;
            font-weight: 500;
        }}
        .hero-block {{
            padding: 0.5rem 0 1.5rem 0;
            margin-bottom: 1rem;
        }}
        .hero-grid {{
            display: grid;
            grid-template-columns: 1.55fr 1fr;
            gap: 2rem;
            align-items: start;
        }}
        @media (max-width: 720px) {{
            .hero-grid {{ grid-template-columns: 1fr; }}
        }}
        .hero-stats {{
            display: flex;
            gap: 2rem;
            margin-top: 0.5rem;
            flex-wrap: wrap;
        }}
        .hero-stat-value {{
            font-family: 'Manrope', sans-serif;
            font-size: 3.4rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.045em;
            color: {PRIMARY_DARK};
        }}
        .hero-stat-unit {{
            font-size: 0.55em;
            font-weight: 700;
            color: {PRIMARY_DARK};
            margin-left: 0.05em;
        }}
        .hero-stat-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {MUTED};
            margin-top: 0.4rem;
            font-weight: 500;
        }}
        .side-stack {{
            position: relative;
            min-height: 240px;
            margin-bottom: 1rem;
        }}
        .model-card {{
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(14px) saturate(160%);
            -webkit-backdrop-filter: blur(14px) saturate(160%);
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1rem 1.2rem;
        }}
        .model-card-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: {PRIMARY};
            margin-bottom: 0.6rem;
        }}
        .model-card-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 0.35rem 0;
            border-bottom: 1px solid {BORDER};
            font-size: 0.78rem;
        }}
        .model-card-row:last-child {{ border-bottom: none; }}
        .model-card-row span:first-child {{ color: {MUTED}; }}
        .model-card-row span:last-child {{
            color: {PRIMARY_DARK};
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
        }}
        .side-card {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(18px) saturate(170%);
            -webkit-backdrop-filter: blur(18px) saturate(170%);
            border: 1px solid rgba(31, 165, 219, 0.18);
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 4px 24px -8px rgba(27, 54, 93, 0.18);
            position: absolute;
            top: 0; left: 0; right: 0;
            opacity: 0;
            animation: cardCycle 18s linear infinite;
        }}
        .side-card:nth-of-type(1) {{ animation-delay: 0s; }}
        .side-card:nth-of-type(2) {{ animation-delay: 6s; }}
        .side-card:nth-of-type(3) {{ animation-delay: 12s; }}
        @keyframes cardCycle {{
            0% {{ opacity: 0; transform: translateY(6px); }}
            3% {{ opacity: 1; transform: translateY(0); }}
            30% {{ opacity: 1; transform: translateY(0); }}
            33% {{ opacity: 0; transform: translateY(-6px); }}
            100% {{ opacity: 0; transform: translateY(-6px); }}
        }}
        .side-card-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: {ACCENT_DARK};
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}
        .side-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: {ACCENT};
            box-shadow: 0 0 0 0 rgba(123, 192, 67, 0.65);
            animation: sideDotPulse 2s ease-out infinite;
        }}
        @keyframes sideDotPulse {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(123, 192, 67, 0.65); }}
            70% {{ box-shadow: 0 0 0 8px rgba(123, 192, 67, 0); }}
        }}
        .side-card-sub {{
            font-size: 0.78rem;
            color: {MUTED};
            margin-bottom: 0.4rem;
        }}
        .side-card-name {{
            font-family: 'Manrope', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: {PRIMARY_DARK};
            margin-bottom: 0.85rem;
            letter-spacing: -0.015em;
        }}
        .side-card-confidence {{
            display: flex;
            align-items: baseline;
            margin-bottom: 0.55rem;
        }}
        .side-card-conf-num {{
            font-family: 'Manrope', sans-serif;
            font-size: 3.2rem;
            font-weight: 800;
            color: {PRIMARY};
            line-height: 1;
            letter-spacing: -0.045em;
        }}
        .side-card-conf-unit {{
            font-size: 1.2rem;
            font-weight: 600;
            color: {PRIMARY};
            margin-left: 0.15rem;
        }}
        .side-card-meta {{
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6rem;
            letter-spacing: 0.12em;
            color: {MUTED};
            padding-top: 0.7rem;
            border-top: 1px solid {BORDER};
            font-weight: 500;
        }}
        /* Apple-style liquid glass button (backdrop-blur + layered inset shadows).
           No SVG displacement — Chrome would show refraction, all browsers
           get clean frosted glass. */
        .glass-btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.6rem 1.3rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(14px) saturate(170%);
            -webkit-backdrop-filter: blur(14px) saturate(170%);
            color: {PRIMARY_DARK};
            font-weight: 600;
            font-size: 0.9rem;
            font-family: 'Manrope', sans-serif;
            letter-spacing: -0.005em;
            text-decoration: none;
            transition: transform 280ms cubic-bezier(0.16, 1, 0.3, 1),
                        box-shadow 280ms cubic-bezier(0.16, 1, 0.3, 1),
                        background 280ms ease;
            box-shadow:
                inset 0 0 0 1px rgba(255, 255, 255, 0.45),
                inset 1.5px 2.5px 0 -2px rgba(255, 255, 255, 0.85),
                inset -2px -2px 0 -2px rgba(255, 255, 255, 0.7),
                inset -0.5px -1px 3px 0 rgba(0, 0, 0, 0.10),
                inset 0 2.5px 3.5px -2px rgba(0, 0, 0, 0.16),
                0 1px 4px 0 rgba(27, 54, 93, 0.08),
                0 5px 14px -2px rgba(27, 54, 93, 0.10);
            cursor: pointer;
        }}
        .glass-btn:hover {{
            transform: scale(1.04);
            background: rgba(255, 255, 255, 0.7);
            box-shadow:
                inset 0 0 0 1px rgba(255, 255, 255, 0.55),
                inset 1.5px 2.5px 0 -2px rgba(255, 255, 255, 0.95),
                inset -2px -2px 0 -2px rgba(255, 255, 255, 0.85),
                0 4px 16px -2px rgba(27, 54, 93, 0.18);
        }}
        .glass-btn:active {{ transform: scale(0.97); }}
        .glass-btn-sm {{
            padding: 0.4rem 0.95rem;
            font-size: 0.82rem;
        }}
        .glass-btn .arrow {{
            font-size: 1rem;
            line-height: 1;
            transition: transform 280ms cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .glass-btn:hover .arrow {{ transform: translateX(2px); }}

        .hero-chip {{
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: {PRIMARY};
            margin-bottom: 1rem;
        }}
        .text-link {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'Manrope', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: {PRIMARY} !important;
            text-decoration: none !important;
            border: none !important;
            transition: gap 0.25s ease;
        }}
        .text-link:hover {{ gap: 0.8rem; }}
        .text-link .arrow {{
            font-size: 1.05rem;
            line-height: 1;
        }}
        .prose {{
            font-family: 'Manrope', sans-serif;
            font-size: 1.02rem;
            line-height: 1.7;
            color: {TEXT_BODY};
            border-left: 3px solid {ACCENT};
            padding-left: 1.25rem;
            margin: 0.5rem 0;
            font-weight: 400;
        }}
        .footer-card {{
            background: linear-gradient(180deg, {CARD_TINT} 0%, {CARD} 100%);
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 2.2rem 2rem 1.5rem 2rem;
            text-align: left;
            margin-top: 3rem;
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
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: {PRIMARY};
            margin-bottom: 0.85rem;
            font-weight: 500;
        }}
        .meta-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 0.4rem 0;
            border-bottom: 1px solid {BORDER};
            font-size: 0.9rem;
        }}
        .meta-row:last-child {{ border-bottom: none; }}
        .meta-key {{
            color: {MUTED};
            font-weight: 500;
        }}
        .meta-val {{
            color: {PRIMARY_DARK};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 500;
            text-align: right;
        }}
        .footer-bar {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            padding-top: 1.25rem;
            border-top: 1px solid {BORDER};
        }}
        @media (max-width: 640px) {{
            .footer-bar {{ flex-direction: column; align-items: flex-start; }}
        }}
        .avatar {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-family: 'Manrope', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.85rem;
            box-shadow: 0 4px 14px rgba(31, 165, 219, 0.25);
            animation: avatarPulse 4s ease-in-out infinite;
        }}
        @keyframes avatarPulse {{
            0%, 100% {{ box-shadow: 0 4px 14px rgba(31, 165, 219, 0.20); }}
            50% {{ box-shadow: 0 6px 22px rgba(31, 165, 219, 0.42); }}
        }}
        .avatar-wrap {{
            width: 72px;
            height: 72px;
            border-radius: 50%;
            overflow: hidden;
            display: inline-block;
            margin-bottom: 0.85rem;
            box-shadow: 0 4px 14px rgba(31, 165, 219, 0.25);
            animation: avatarPulse 4s ease-in-out infinite;
            border: 3px solid {CARD};
        }}
        .avatar-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            /* translate +X moves image RIGHT in wrapper, so face appears
               LEFT in the visible circle. +Y moves image DOWN, face appears UP. */
            transform: scale(1.55) translate(20%, 22%);
            transform-origin: center;
            display: block;
        }}
        .footer-name {{
            font-family: 'Manrope', sans-serif;
            font-size: 1.05rem;
            color: {PRIMARY_DARK};
            margin-bottom: 0.2rem;
            font-weight: 700;
        }}
        .footer-bio {{ color: {MUTED}; font-size: 0.82rem; }}
        .footer-links {{ display: flex; gap: 1.1rem; }}
        .footer-links a {{
            color: {PRIMARY};
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .footer-links a:hover {{ text-decoration: underline; }}
        .copyright {{
            color: {MUTED};
            font-size: 0.72rem;
            margin-top: 1.25rem;
            padding-top: 1rem;
            border-top: 1px solid {BORDER};
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.04em;
        }}
        section[data-testid="stSidebar"] {{ display: none; }}
        .block-container {{ max-width: 780px; padding-top: 4rem; }}

        .method-card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-left: 3px solid {ACCENT};
            border-radius: 10px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            height: 100%;
            transition: box-shadow 0.18s ease, transform 0.18s ease, border-left-color 0.18s ease;
        }}
        .method-card:hover {{
            box-shadow: 0 6px 18px rgba(27, 54, 93, 0.08);
            transform: translateY(-2px);
            border-left-color: {PRIMARY};
        }}
        .method-title {{
            font-family: 'Manrope', sans-serif;
            font-size: 1rem;
            color: {PRIMARY_DARK};
            margin-bottom: 0.5rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        .method-body {{
            font-size: 0.92rem;
            color: {TEXT_BODY};
            line-height: 1.6;
            font-weight: 400;
        }}
        .section-note {{
            color: {MUTED};
            font-size: 0.92rem;
            line-height: 1.6;
            margin: 0.25rem 0 1.5rem 0;
            max-width: 62ch;
        }}
        .cm-label {{
            font-family: 'Manrope', sans-serif;
            color: {PRIMARY_DARK};
            font-size: 0.95rem;
            text-align: center;
            margin-bottom: 0.4rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        /* WDW-style section labels: small mono uppercase before serif title */
        .sec-num {{
            display: block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            color: {PRIMARY};
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin: 2.5rem 0 0.6rem 0;
        }}
        .stApp h3 {{
            margin-top: 0 !important;
            padding-bottom: 1rem;
            border-bottom: 1px solid {BORDER};
        }}

        /* Sticky top nav with glass-blur backdrop */
        .top-nav {{
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 0.4rem;
            padding: 0.7rem 1rem;
            margin: -2.5rem -1rem 1.5rem -1rem;
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(18px) saturate(180%);
            -webkit-backdrop-filter: blur(18px) saturate(180%);
            border-bottom: 1px solid {BORDER};
        }}
        .top-nav .brand {{
            font-family: 'Fraunces', serif;
            font-weight: 500;
            color: {PRIMARY_DARK};
            font-size: 0.95rem;
            margin-right: auto;
            padding-left: 0.3rem;
        }}
        .top-nav a {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: {MUTED};
            text-decoration: none;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            transition: color 200ms ease, background 200ms ease;
            font-weight: 500;
        }}
        .top-nav a:hover {{
            color: {PRIMARY_DARK};
            background: rgba(31, 165, 219, 0.08);
        }}

        /* Expander styling: premium tinted card look */
        [data-testid="stExpander"] {{
            background: {CARD_TINT};
            border: 1px solid {BORDER};
            border-left: 3px solid {ACCENT};
            border-radius: 10px;
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 3px rgba(27, 54, 93, 0.03);
            transition: box-shadow 0.18s ease, border-left-color 0.18s ease;
        }}
        [data-testid="stExpander"]:hover {{
            box-shadow: 0 4px 12px rgba(27, 54, 93, 0.07);
            border-left-color: {PRIMARY};
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details > summary {{
            font-weight: 600 !important;
            color: {PRIMARY_DARK} !important;
            padding: 0.4rem 0 !important;
        }}
        [data-testid="stExpander"] details[open] {{
            background: transparent;
        }}

        /* Radio buttons: chip-style premium */
        .stRadio > div {{ gap: 0.5rem !important; }}
        .stRadio label {{
            background: {CARD};
            border: 1.5px solid {BORDER};
            padding: 0.45rem 1rem !important;
            border-radius: 8px;
            transition: all 0.18s ease;
            cursor: pointer;
        }}
        .stRadio label:hover {{
            border-color: {PRIMARY};
            background: {CARD_TINT};
            transform: translateY(-1px);
        }}
        .stRadio label[data-checked="true"] {{
            border-color: {PRIMARY};
            background: {CARD_TINT};
        }}

        /* Selectbox: subtle premium frame */
        .stSelectbox [data-baseweb="select"] > div {{
            border-radius: 8px !important;
            border-color: {BORDER} !important;
            background: {CARD} !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }}
        .stSelectbox [data-baseweb="select"] > div:hover {{
            border-color: {PRIMARY} !important;
        }}
        .stSelectbox [data-baseweb="select"] > div:focus-within {{
            border-color: {PRIMARY} !important;
            box-shadow: 0 0 0 3px rgba(31, 165, 219, 0.12);
        }}

        /* File uploader */
        [data-testid="stFileUploader"] section {{
            background: {CARD_TINT};
            border: 1.5px dashed {BORDER};
            border-radius: 10px;
        }}

        /* Streamlit alert boxes: cleaner */
        [data-testid="stAlert"] {{
            border-radius: 10px;
            border: 1px solid {BORDER};
            padding: 0.85rem 1rem !important;
        }}

        /* Custom divider replacement (subtle gradient line) */
        .gradient-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, {BORDER} 20%, {BORDER} 80%, transparent 100%);
            margin: 2rem 0;
            border: none;
        }}

        /* Streamlit alerts */
        [data-testid="stAlert"] {{
            border-radius: 10px;
            animation: smoothFade 0.35s ease-out;
        }}

        /* Smooth fade on plots, images, and prediction renders so switching
           between samples doesn't snap-cut. No animation-fill-mode set,
           so default opacity is 1 if animation never fires. */
        @keyframes smoothFade {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        [data-testid="stPyplotChart"],
        [data-testid="stPyplotChart"] img,
        [data-testid="stImage"],
        [data-testid="stImage"] img {{
            animation: smoothFade 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .pred-reveal {{
            animation: smoothFade 0.42s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        /* Entry animations — safe pattern: animation-fill-mode: backwards
           means elements default to visible if animation fails, only invisible
           during the delay (which is brief). No catch-all selectors. */
        @keyframes riseIn {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .hero-block {{
            animation: riseIn 0.55s cubic-bezier(0.16, 1, 0.3, 1) backwards;
        }}
        [data-testid="stMetric"] {{
            animation: riseIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) backwards;
        }}
        [data-testid="stMetric"]:nth-of-type(1) {{ animation-delay: 80ms; }}
        [data-testid="stMetric"]:nth-of-type(2) {{ animation-delay: 130ms; }}
        [data-testid="stMetric"]:nth-of-type(3) {{ animation-delay: 180ms; }}
        [data-testid="stMetric"]:nth-of-type(4) {{ animation-delay: 230ms; }}
        [data-testid="stMetric"]:nth-of-type(5) {{ animation-delay: 280ms; }}
        [data-testid="stMetric"]:nth-of-type(6) {{ animation-delay: 330ms; }}
        .method-card {{
            animation: riseIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) backwards;
        }}
        .method-card:nth-of-type(1) {{ animation-delay: 60ms; }}
        .method-card:nth-of-type(2) {{ animation-delay: 120ms; }}
        .method-card:nth-of-type(3) {{ animation-delay: 180ms; }}
        .method-card:nth-of-type(4) {{ animation-delay: 240ms; }}
        .footer-card {{
            animation: riseIn 0.55s cubic-bezier(0.16, 1, 0.3, 1) backwards;
            animation-delay: 100ms;
        }}

        /* Scroll-triggered reveal for content below the fold. Uses CSS view()
           timeline (Chrome/Edge 115+, Safari 26+). Older browsers ignore
           animation-timeline and the animation plays once on load instead —
           still safe, just not scroll-driven. */
        @keyframes scrollRise {{
            from {{ opacity: 0; transform: translateY(40px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        [data-testid="stPyplotChart"],
        [data-testid="stImage"],
        [data-testid="stExpander"],
        .method-card,
        .prose,
        .pred-reveal {{
            animation: scrollRise linear both;
            animation-timeline: view();
            animation-range: entry 0% cover 28%;
        }}

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


def render_nav() -> None:
    st.markdown(
        """
        <div class="top-nav">
            <a href="#classify">Classify</a>
            <a href="#compare">Compare</a>
            <a href="#perclass">Classes</a>
            <a href="#method">Method</a>
            <a href="#why">Why</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(comparison: dict, labels: dict) -> None:
    cnn = comparison.get("cnn_1d", {})
    sample_cls = 5
    sample_name = labels.get(sample_cls, f"class_{sample_cls}")
    sample_conf = 64.3

    st.markdown(
        f"""
        <div class="hero-block hero-grid">
            <div class="hero-left">
                <div class="hero-chip">Machine Learning · Biosignals</div>
                <h1 style="margin: 0 0 0.7rem 0;">EMG Hand Gesture Classifier</h1>
                <div style="font-size: 1.05rem; color: {TEXT_BODY}; line-height: 1.55; max-width: 52ch; margin-bottom: 1.5rem;">
                    A study of 1.2 million surface-EMG windows from 27 subjects, classifying 53 hand gestures with a Random Forest baseline and a 1D CNN.
                </div>
                <div class="hero-stats">
                    <div class="hero-stat">
                        <div class="hero-stat-value">1.2<span class="hero-stat-unit">M</span></div>
                        <div class="hero-stat-label">Windows trained</div>
                    </div>
                    <div class="hero-stat">
                        <div class="hero-stat-value">27</div>
                        <div class="hero-stat-label">Subjects</div>
                    </div>
                    <div class="hero-stat">
                        <div class="hero-stat-value">53</div>
                        <div class="hero-stat-label">Gestures</div>
                    </div>
                </div>
                <div class="byline" style="margin: 1.4rem 0 1.2rem 0;">by Johnny Nguyen · UCF Data Science</div>
                <a href="{GITHUB_URL}" target="_blank" class="text-link">
                    View source on GitHub <span class="arrow">→</span>
                </a>
            </div>
            <div class="hero-side">
                <div class="side-stack">
                    <div class="side-card">
                        <div class="side-card-label"><span class="side-dot"></span>Sample · Preview</div>
                        <div class="side-card-sub">Predicted gesture for class 5</div>
                        <div class="side-card-name">{labels.get(5, 'Ring flexion')}</div>
                        <div class="side-card-confidence">
                            <span class="side-card-conf-num">64.3</span><span class="side-card-conf-unit">%</span>
                        </div>
                        <div class="side-card-meta"><span>CONFIDENCE</span><span>BASIC FINGER MOVEMENT</span></div>
                    </div>
                    <div class="side-card">
                        <div class="side-card-label"><span class="side-dot"></span>Sample · Preview</div>
                        <div class="side-card-sub">Predicted gesture for class 25</div>
                        <div class="side-card-name">{labels.get(25, 'Wrist flexion')}</div>
                        <div class="side-card-confidence">
                            <span class="side-card-conf-num">16.3</span><span class="side-card-conf-unit">%</span>
                        </div>
                        <div class="side-card-meta"><span>CONFIDENCE</span><span>HAND CONFIGURATION</span></div>
                    </div>
                    <div class="side-card">
                        <div class="side-card-label"><span class="side-dot"></span>Sample · Preview</div>
                        <div class="side-card-sub">Predicted gesture for class 42</div>
                        <div class="side-card-name">{labels.get(42, 'Tripod grasp')}</div>
                        <div class="side-card-confidence">
                            <span class="side-card-conf-num">12.5</span><span class="side-card-conf-unit">%</span>
                        </div>
                        <div class="side-card-meta"><span>CONFIDENCE</span><span>FUNCTIONAL GRASP</span></div>
                    </div>
                </div>
                <div class="model-card">
                    <div class="model-card-label">Model</div>
                    <div class="model-card-row"><span>Architecture</span><span>1D CNN</span></div>
                    <div class="model-card-row"><span>Parameters</span><span>60,309</span></div>
                    <div class="model-card-row"><span>Test accuracy</span><span>{cnn.get('accuracy', 0):.1%}</span></div>
                </div>
            </div>
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
    st.markdown('<span class="sec-num" id="classify">§ 01 — Try the model</span><h3>Live Classification</h3>', unsafe_allow_html=True)
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
        st.markdown(
            f'<div class="pred-reveal">'
            f'<div style="color:{MUTED}; font-size: 0.75rem; '
            f'text-transform: uppercase; letter-spacing: 0.08em; '
            f'font-weight: 600; margin-bottom: 0.4rem;">Prediction</div>'
            f'<div style="font-family: \'Manrope\', sans-serif; font-size: 1.5rem; '
            f'font-weight: 700; color: {PRIMARY_DARK}; line-height: 1.15; '
            f'margin-bottom: 0.5rem; letter-spacing: -0.01em;">{pred_name}</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; '
            f'font-size: 1.1rem; color: {PRIMARY}; margin-bottom: 0.25rem;">'
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
    st.markdown('<span class="sec-num">§ 02 — Ranked confidence</span><h3>Top 5 Predictions</h3>', unsafe_allow_html=True)
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
    st.markdown('<span class="sec-num" id="compare">§ 03 — Head to head</span><h3>Model Comparison</h3>', unsafe_allow_html=True)
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

    st.markdown('<span class="sec-num" id="perclass">§ 04 — Where the model breaks</span><h3>Per-Class Performance</h3>', unsafe_allow_html=True)
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
    st.markdown('<span class="sec-num" id="method">§ 05 — How it works</span><h3>Methodology</h3>', unsafe_allow_html=True)
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

    render_nav()
    render_hero(comparison, labels)
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
    render_footer(comparison)


if __name__ == "__main__":
    main()
