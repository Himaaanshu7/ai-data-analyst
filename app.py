"""
app.py - DataLens AI — AI-Powered Data Analysis Assistant
Built by Himanshu Mishra
"""

import io
import os

import pandas as pd
import streamlit as st
from groq import Groq

from ai_insights import generate_data_story, generate_insights
from data_analysis import compute_feature_importance, detect_anomalies, profile_dataset
from utils import (
    build_text_report,
    column_type_summary,
    compute_dataset_quality_score,
    df_to_csv_bytes,
    get_score_color,
    get_score_label,
    safe_sample_df,
)
from visualization import (
    plot_boxplots,
    plot_categorical_distributions,
    plot_correlation_heatmap,
    plot_correlation_network,
    plot_feature_importance,
    plot_histograms,
    plot_missing_values,
    plot_quality_radar,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ═══════════════════════ SIDEBAR ═══════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f172a 0%, #1e1b4b 55%, #0c0a1e 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }
[data-testid="stSidebar"] input[type="password"],
[data-testid="stSidebar"] input[type="text"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1.5px solid rgba(99,102,241,0.4) !important;
    border-radius: 8px !important; color: #fff !important;
}
[data-testid="stSidebar"] input::placeholder { color: rgba(255,255,255,0.3) !important; }

/* File uploader custom look */
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
    background: radial-gradient(ellipse at 60% 35%,
        rgba(79,70,229,0.2) 0%, rgba(6,182,212,0.07) 55%, transparent 100%) !important;
    border: 2px dashed rgba(99,102,241,0.65) !important;
    border-radius: 0 0 14px 14px !important;
    padding: 18px 12px 16px !important;
    text-align: center !important;
    transition: border-color 0.25s, background 0.25s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(99,102,241,1) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 0.78rem !important; padding: 7px 20px !important;
    box-shadow: 0 3px 12px rgba(79,70,229,0.5) !important;
    margin-top: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button:hover { opacity: 0.82 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] p {
    color: rgba(165,180,252,0.7) !important; font-size: 0.78rem !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(16,185,129,0.12) !important;
    border-color: rgba(16,185,129,0.35) !important; border-radius: 8px !important;
}

/* ═══════════════════════ HERO ═══════════════════════ */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 42%, #0c4a6e 100%);
    border-radius: 22px; padding: 68px 56px 60px;
    text-align: center; position: relative; overflow: hidden; margin-bottom: 28px;
}
.hero::before {
    content:''; position:absolute; inset:0;
    background:
        radial-gradient(ellipse at 18% 55%, rgba(99,102,241,0.22) 0%, transparent 52%),
        radial-gradient(ellipse at 82% 45%, rgba(6,182,212,0.16) 0%, transparent 52%),
        radial-gradient(ellipse at 50% 90%, rgba(124,58,237,0.1) 0%, transparent 40%);
}
.hero-badge {
    position:relative; display:inline-flex; align-items:center; gap:6px;
    background:rgba(99,102,241,0.2); border:1px solid rgba(99,102,241,0.5);
    border-radius:100px; padding:5px 18px; font-size:0.72rem; font-weight:700;
    color:#a5b4fc; letter-spacing:2px; text-transform:uppercase; margin-bottom:20px;
}
.hero-title {
    position:relative; font-size:4.2rem; font-weight:900; line-height:1.05; margin:0 0 16px;
    color:#ffffff; text-shadow:0 2px 24px rgba(99,102,241,0.4);
}
.hero-sub { position:relative; color:#94a3b8; font-size:1.1rem; max-width:560px; margin:0 auto 34px; line-height:1.65; }
.hero-chips { position:relative; display:flex; justify-content:center; flex-wrap:wrap; gap:10px; }
.chip {
    background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.14);
    border-radius:8px; padding:7px 16px; color:#cbd5e1; font-size:0.84rem; font-weight:500;
}

/* Steps */
.steps { display:flex; border-radius:12px; overflow:hidden; margin:8px 0 4px; }
.step {
    flex:1; background:#fff; border:1px solid #e2e8f0;
    border-left:none; padding:18px 14px; text-align:center;
}
.step:first-child { border-left:1px solid #e2e8f0; border-radius:12px 0 0 12px; }
.step:last-child  { border-radius:0 12px 12px 0; }
.step-num  { font-size:1.4rem; font-weight:900; color:#4f46e5; line-height:1; }
.step-text { font-size:0.76rem; color:#475569; margin-top:5px; font-weight:500; line-height:1.45; }

/* Feature cards */
.feat-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin:8px 0; }
.feat-card {
    background:#fff; border-radius:14px; padding:22px;
    border:1px solid #e8eaf6; box-shadow:0 2px 12px rgba(79,70,229,0.05);
    transition:box-shadow 0.2s,transform 0.2s;
}
.feat-card:hover { box-shadow:0 8px 30px rgba(79,70,229,0.13); transform:translateY(-2px); }
.feat-icon  { font-size:1.7rem; margin-bottom:8px; }
.feat-title { font-size:0.92rem; font-weight:700; color:#1e293b; margin-bottom:5px; }
.feat-desc  { font-size:0.79rem; color:#64748b; line-height:1.6; }

/* ═══════════════════════ DATASET BANNER ═══════════════════════ */
.ds-banner {
    background:linear-gradient(135deg,#1e1b4b 0%,#312e81 52%,#1e40af 100%);
    border-radius:16px; padding:22px 28px; color:#fff;
    display:flex; align-items:center; gap:18px; margin-bottom:14px;
}
.ds-icon { font-size:2.2rem; }
.ds-name { font-size:1.3rem; font-weight:800; letter-spacing:-0.3px; }
.ds-meta { font-size:0.8rem; color:#a5b4fc; margin-top:3px; }

/* ═══════════════════════ METRIC CARDS ═══════════════════════ */
.metric-row { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin:12px 0; }
.mcard {
    background:#fff; border-radius:12px; padding:15px 10px 12px;
    text-align:center; border:1px solid #e8eaf6;
    box-shadow:0 2px 8px rgba(15,23,42,0.05);
    border-top:3px solid var(--accent,#4f46e5); transition:box-shadow 0.2s;
}
.mcard:hover { box-shadow:0 6px 20px rgba(79,70,229,0.1); }
.mcard .mv { font-size:1.4rem; font-weight:800; color:#0f172a; line-height:1; }
.mcard .ml { font-size:0.68rem; font-weight:600; color:#64748b; margin-top:4px;
             text-transform:uppercase; letter-spacing:0.6px; }

/* ═══════════════════════ QUALITY PILL ═══════════════════════ */
.quality-pill {
    border-radius:14px; padding:14px 26px; color:#fff;
    display:flex; align-items:center; gap:20px; margin:4px 0 16px;
}
.qp-green  { background:linear-gradient(135deg,#059669,#10b981); box-shadow:0 4px 22px rgba(16,185,129,0.28); }
.qp-orange { background:linear-gradient(135deg,#d97706,#f59e0b); box-shadow:0 4px 22px rgba(245,158,11,0.28); }
.qp-red    { background:linear-gradient(135deg,#dc2626,#ef4444); box-shadow:0 4px 22px rgba(239,68,68,0.28); }
.qp-score  { font-size:2.5rem; font-weight:900; line-height:1; }
.qp-label  { font-size:1rem; font-weight:700; }
.qp-detail { font-size:0.78rem; opacity:0.88; margin-top:2px; }

/* ═══════════════════════ CHART CARD ═══════════════════════ */
.chart-card {
    background:#fff; border-radius:16px; padding:0;
    border:1px solid #e8eaf6;
    box-shadow:0 2px 16px rgba(15,23,42,0.06);
    margin-bottom:16px; overflow:hidden;
}
.chart-card-header {
    padding:12px 20px 10px;
    background:linear-gradient(135deg,#f5f3ff 0%,#eef2ff 100%);
    border-bottom:1px solid #e0e7ff;
    display:flex; align-items:center; gap:10px;
}
.chart-card-icon  { font-size:1.15rem; }
.chart-card-title { font-size:0.92rem; font-weight:800; color:#1e1b4b; letter-spacing:-0.2px; }
.chart-card-sub   { font-size:0.72rem; color:#6366f1; margin-left:auto; font-weight:500; }
.chart-card-body  { padding:4px 4px 8px; }

/* ═══════════════════════ RECOMMENDATIONS ═══════════════════════ */
.rec-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:12px; margin:10px 0; }
.rec-card {
    border-radius:12px; padding:14px 16px;
    display:flex; align-items:flex-start; gap:12px;
    border-left:4px solid var(--rc,#4f46e5);
    background:var(--rbg,#f0f0ff);
}
.rec-icon  { font-size:1.3rem; flex-shrink:0; margin-top:2px; }
.rec-title { font-size:0.83rem; font-weight:700; color:#1e293b; margin-bottom:2px; }
.rec-body  { font-size:0.78rem; color:#475569; line-height:1.55; }

/* ═══════════════════════ SECTION HEADS ═══════════════════════ */
.sec-head {
    display:flex; align-items:center; gap:10px;
    font-size:1.1rem !important; font-weight:800 !important;
    color:#a5b4fc !important;
    margin:24px 0 10px; padding-left:14px;
    border-left:4px solid #6366f1;
    background:linear-gradient(90deg,rgba(99,102,241,0.12) 0%,transparent 100%);
    border-radius:0 8px 8px 0; padding-top:8px; padding-bottom:8px;
}
.sec-head * { color:#a5b4fc !important; }

/* ═══════════════════════ AI CARDS ═══════════════════════ */
.insight-card {
    background:linear-gradient(145deg,#0f172a 0%,#1e1b4b 100%);
    border:1px solid rgba(99,102,241,0.3); border-radius:16px;
    padding:28px 32px; color:#e2e8f0; line-height:1.8; font-size:0.93rem;
    box-shadow:0 0 45px rgba(99,102,241,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
}
.insight-card h1,.insight-card h2,.insight-card h3 { color:#a5b4fc !important; margin-top:20px; }
.insight-card strong { color:#c7d2fe !important; }
.insight-card li { margin-bottom:5px; }
.insight-card code { background:rgba(255,255,255,0.1); padding:2px 7px; border-radius:5px; font-size:0.88em; }
.insight-card hr { border-color:rgba(255,255,255,0.08); }

.story-card {
    background:linear-gradient(135deg,#fffbeb 0%,#fff7ed 100%);
    border-left:5px solid #f59e0b; border-radius:0 16px 16px 0;
    padding:30px 36px; color:#1c1917; line-height:1.92; font-size:1.04rem;
    font-style:italic; box-shadow:0 4px 24px rgba(245,158,11,0.1);
}

/* ═══════════════════════ EMPTY STATE ═══════════════════════ */
.empty-state {
    background:#f8fafc; border:2px dashed #cbd5e1; border-radius:14px;
    padding:44px 28px; text-align:center; color:#64748b;
}
.empty-state .es-icon  { font-size:2.5rem; margin-bottom:12px; }
.empty-state .es-title { font-size:0.98rem; font-weight:700; color:#334155; margin-bottom:4px; }
.empty-state .es-body  { font-size:0.83rem; }

/* ═══════════════════════ TABS ═══════════════════════ */
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:2px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] {
    border-radius:10px 10px 0 0; font-weight:600;
    font-size:0.88rem; padding:10px 22px; color:#64748b;
}
.stTabs [aria-selected="true"] { background:#ede9fe !important; color:#4f46e5 !important; }

/* ═══════════════════════ BUTTONS ═══════════════════════ */
.stDownloadButton > button {
    background:linear-gradient(135deg,#4f46e5,#6366f1) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; box-shadow:0 3px 12px rgba(79,70,229,0.3) !important;
}
.stDownloadButton > button:hover { opacity:0.87 !important; }
button[data-testid="baseButton-primary"] {
    background:linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    border:none !important; border-radius:10px !important; font-weight:700 !important;
    box-shadow:0 4px 16px rgba(79,70,229,0.38) !important;
}

/* ═══════════════════════ VIZ DASHBOARD ═══════════════════════ */
.viz-section-label {
    font-size:0.72rem; font-weight:800; letter-spacing:2.5px; text-transform:uppercase;
    color:#4f46e5; margin:26px 0 8px; padding-left:14px;
    border-left:3px solid #4f46e5;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_df(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def _run_analysis(file_bytes: bytes) -> tuple:
    df         = pd.read_csv(io.BytesIO(file_bytes))
    profile    = profile_dataset(df)
    quality    = compute_dataset_quality_score(df)
    anomalies  = detect_anomalies(df)
    importance = compute_feature_importance(df)
    return profile, quality, anomalies, importance


def _chart_card(icon: str, title: str, subtitle: str = "") -> None:
    """Render a chart card header (call before st.plotly_chart)."""
    sub_html = f'<span class="chart-card-sub">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div class="chart-card-header">'
        f'  <span class="chart-card-icon">{icon}</span>'
        f'  <span class="chart-card-title">{title}</span>'
        f'  {sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _smart_recommendations(profile: dict, quality: dict, anomalies: dict) -> list[dict]:
    recs = []
    heavy_missing = {c: i for c, i in profile["missing_per_column"].items() if i["percent"] > 20}
    if heavy_missing:
        cols_str = ", ".join(f"`{c}`" for c in list(heavy_missing.keys())[:3])
        recs.append(dict(icon="🔧", title="Handle Missing Data",
            body=f"{len(heavy_missing)} column(s) exceed 20% missing ({cols_str}). "
                 "Consider imputation or dropping low-coverage columns.",
            color="#d97706", bg="#fffbeb"))
    if profile["duplicate_rows"] > 0:
        pct = round(profile["duplicate_rows"] / profile["rows"] * 100, 1)
        recs.append(dict(icon="🔁", title="Remove Duplicates",
            body=f"{profile['duplicate_rows']:,} duplicate rows ({pct}%). "
                 "Run `df.drop_duplicates()` before any modelling.",
            color="#7c3aed", bg="#f5f3ff"))
    n_hc = len(profile.get("high_correlations", []))
    if n_hc >= 2:
        recs.append(dict(icon="🔗", title="Multicollinearity Alert",
            body=f"{n_hc} strongly correlated feature pairs (|r| ≥ 0.7). "
                 "Consider PCA or VIF analysis before regression.",
            color="#0891b2", bg="#ecfeff"))
    bad = [c for c, i in anomalies.items() if i["percent"] > 5]
    if bad:
        recs.append(dict(icon="⚠️", title="Outlier Treatment",
            body=f"{len(bad)} column(s) have >5% outliers. "
                 "Use RobustScaler or Winsorization before modelling.",
            color="#dc2626", bg="#fff1f2"))
    binary_cols = [c for c in profile["categorical_col_names"]
                   if len(profile["categorical_value_counts"].get(c, {})) == 2]
    if binary_cols:
        recs.append(dict(icon="🎯", title="Potential Classification Target",
            body=f"Binary column(s): {', '.join(f'`{c}`' for c in binary_cols[:3])}. "
                 "Could be used directly as a classification label.",
            color="#059669", bg="#f0fdf4"))
    stats = profile.get("descriptive_stats")
    if stats is not None and "std" in stats.index:
        zero_var = [c for c in stats.columns if stats.loc["std", c] == 0]
        if zero_var:
            recs.append(dict(icon="📉", title="Zero-Variance Columns",
                body=f"`{'`, `'.join(zero_var)}` carry no signal — constant columns. Drop them.",
                color="#64748b", bg="#f8fafc"))
    if not recs:
        recs.append(dict(icon="🚀", title="Dataset Looks Great!",
            body=f"Quality {quality['overall_score']}/100 — no major issues found. "
                 "Ready for feature engineering or modelling.",
            color="#059669", bg="#f0fdf4"))
    return recs


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:26px 0 18px;">
        <div style="font-size:3rem;line-height:1;filter:drop-shadow(0 0 14px rgba(99,102,241,0.6));">🔬</div>
        <div style="font-size:1.2rem;font-weight:900;letter-spacing:-0.5px;margin-top:10px;">DataLens AI</div>
        <div style="font-size:0.65rem;color:rgba(165,180,252,0.6);letter-spacing:2px;
                    text-transform:uppercase;margin-top:4px;">AI Data Analysis Tool</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── API key: secrets.toml → env var → manual input ─────────────────────
    _secrets_key = st.secrets.get("GROQ_API_KEY", "")
    _env_key     = os.environ.get("GROQ_API_KEY", "")
    _auto_key    = _secrets_key or _env_key

    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:1.8px;'
                'text-transform:uppercase;color:rgba(165,180,252,0.6);margin-bottom:5px;">'
                '🔑 &nbsp;GROQ API KEY</p>', unsafe_allow_html=True)

    if _auto_key:
        masked = _auto_key[:8] + "·" * 8 + _auto_key[-4:]
        st.markdown(
            f'<div style="background:rgba(16,185,129,0.14);border:1px solid '
            f'rgba(16,185,129,0.38);border-radius:8px;padding:8px 12px;font-size:0.78rem;">'
            f'<div style="font-weight:700;color:#6ee7b7;">✅ &nbsp;Key loaded automatically</div>'
            f'<div style="color:rgba(165,180,252,0.55);font-size:0.68rem;margin-top:3px;">'
            f'<code style="background:transparent;">{masked}</code></div></div>',
            unsafe_allow_html=True,
        )
        api_key = _auto_key
    else:
        api_key = st.text_input("API Key", type="password", placeholder="gsk_...",
                                label_visibility="collapsed")
        if api_key:
            st.markdown('<div style="background:rgba(16,185,129,0.14);border:1px solid '
                        'rgba(16,185,129,0.38);border-radius:8px;padding:6px 12px;font-size:0.78rem;">'
                        '✅ &nbsp;API key configured</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:rgba(245,158,11,0.12);border:1px solid '
                'rgba(245,158,11,0.4);border-radius:8px;padding:7px 12px;font-size:0.74rem;'
                'color:rgba(253,230,138,0.85);">'
                '⚡ Paste your Groq key above, or add it to<br>'
                '<code style="background:transparent;font-size:0.72rem;">'
                '.streamlit/secrets.toml</code> to auto-load.</div>',
                unsafe_allow_html=True,
            )
    st.markdown("---")

    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:1.8px;'
                'text-transform:uppercase;color:rgba(165,180,252,0.6);margin-bottom:0;">'
                '📂 &nbsp;UPLOAD DATASET</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(79,70,229,0.16),rgba(6,182,212,0.07));
                border:1.5px dashed rgba(99,102,241,0.55);border-bottom:none;
                border-radius:14px 14px 0 0;padding:16px 12px 10px;text-align:center;
                margin-top:8px;">
        <div style="font-size:2rem;line-height:1;filter:drop-shadow(0 2px 8px rgba(99,102,241,0.4));">📁</div>
        <div style="font-weight:700;font-size:0.84rem;color:#e2e8f0;margin:7px 0 3px;">
            Drag &amp; drop your CSV
        </div>
        <div style="font-size:0.68rem;color:rgba(165,180,252,0.55);">
            .csv files &nbsp;·&nbsp; up to 50 MB
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"],
                                     label_visibility="collapsed")

    if uploaded_file:
        size_kb  = round(uploaded_file.size / 1024, 1)
        size_str = f"{size_kb} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        st.markdown(
            f'<div style="background:rgba(79,70,229,0.14);border:1.5px solid '
            f'rgba(99,102,241,0.35);border-radius:0 0 12px 12px;border-top:none;'
            f'padding:9px 14px;">'
            f'<div style="font-size:0.8rem;font-weight:700;color:#c7d2fe;">'
            f'✅ &nbsp;{uploaded_file.name}</div>'
            f'<div style="font-size:0.68rem;color:rgba(165,180,252,0.6);margin-top:2px;">'
            f'Size: {size_str}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")

    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:1.8px;'
                'text-transform:uppercase;color:rgba(165,180,252,0.6);margin-bottom:6px;">'
                '⚙️ &nbsp;OPTIONS</p>', unsafe_allow_html=True)
    preview_rows = st.slider("Preview rows", 5, 50, 10)
    enable_story = st.checkbox("Enable Data Story", value=True)
    st.markdown("---")

    st.markdown("""
    <div style="text-align:center;padding:4px 0;">
        <div style="font-size:0.8rem;font-weight:700;color:rgba(165,180,252,0.9);">
            Built by Himanshu Mishra
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ════════════════════════════════════════════════════════════════════════════════
if uploaded_file is None:
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🔬 &nbsp; AI-POWERED ANALYTICS</div>
        <div class="hero-title">DataLens AI</div>
        <div class="hero-sub">Drop any CSV dataset — get instant exploratory analysis,
            interactive dashboards, and AI-powered insights.</div>
        <div class="hero-chips">
            <span class="chip">📋 Auto Profiling</span>
            <span class="chip">📈 Interactive Charts</span>
            <span class="chip">🕸 Correlation Network</span>
            <span class="chip">🎯 Smart Recommendations</span>
            <span class="chip">🤖 Claude Insights</span>
            <span class="chip">📖 Data Story</span>
            <span class="chip">🏅 Quality Score</span>
            <span class="chip">📥 Export Report</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">How it works</div>', unsafe_allow_html=True)
    st.markdown("""<div class="steps">
        <div class="step"><div class="step-num">1</div><div class="step-text">Paste Anthropic API key in sidebar</div></div>
        <div class="step"><div class="step-num">2</div><div class="step-text">Upload any CSV dataset</div></div>
        <div class="step"><div class="step-num">3</div><div class="step-text">Instant EDA &amp; smart recommendations</div></div>
        <div class="step"><div class="step-num">4</div><div class="step-text">One-click Claude insights &amp; data story</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">What you get</div>', unsafe_allow_html=True)
    st.markdown("""<div class="feat-grid">
        <div class="feat-card"><div class="feat-icon">📋</div>
            <div class="feat-title">Automatic Data Profiling</div>
            <div class="feat-desc">Shape · types · missing values · descriptive stats · correlations · duplicates</div></div>
        <div class="feat-card"><div class="feat-icon">📈</div>
            <div class="feat-title">Interactive Dashboard</div>
            <div class="feat-desc">Histograms · heatmap · correlation network · boxplots · categorical charts · feature importance</div></div>
        <div class="feat-card"><div class="feat-icon">🎯</div>
            <div class="feat-title">Smart Recommendations</div>
            <div class="feat-desc">Rule-based action cards: missing data, outliers, multicollinearity, zero-variance, potential targets</div></div>
        <div class="feat-card"><div class="feat-icon">🤖</div>
            <div class="feat-title">Claude AI Insights + Story</div>
            <div class="feat-desc">Key findings · trends · anomalies · business questions · plain-English narrative</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Sample datasets to try</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.info("🚢 **Titanic** — Mixed types, missing values, survival patterns")
    s2.info("🌸 **Iris** — Clean numeric-only, strong inter-feature correlations")
    s3.info("💰 **Tips** — Real-world dataset with tip/bill relationships")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# LOAD + ANALYSE
# ════════════════════════════════════════════════════════════════════════════════
file_bytes = uploaded_file.read()
with st.spinner("Profiling dataset …"):
    df                                      = _load_df(file_bytes)
    profile, quality, anomalies, importance = _run_analysis(file_bytes)

score     = quality["overall_score"]
label     = get_score_label(score)
color_key = get_score_color(score)

# Dataset banner
st.markdown(
    f'<div class="ds-banner"><span class="ds-icon">📊</span><div>'
    f'<div class="ds-name">{uploaded_file.name}</div>'
    f'<div class="ds-meta">{profile["rows"]:,} rows &nbsp;·&nbsp; {profile["columns"]} columns'
    f' &nbsp;·&nbsp; {profile["memory_usage_mb"]} MB in memory</div>'
    f'</div></div>', unsafe_allow_html=True)

# Metric cards
_accents = ["#4f46e5","#0891b2","#059669","#d97706","#dc2626","#7c3aed"]
_mlabels = ["Rows","Columns","Numeric","Categorical","Missing","Duplicates"]
_mvals   = [f"{profile['rows']:,}", profile["columns"], profile["numeric_columns"],
            profile["categorical_columns"], f"{quality['missing_cells']:,}", f"{quality['duplicate_rows']:,}"]
cards = '<div class="metric-row">'
for v, l, a in zip(_mvals, _mlabels, _accents):
    cards += f'<div class="mcard" style="--accent:{a};"><div class="mv">{v}</div><div class="ml">{l}</div></div>'
cards += "</div>"
st.markdown(cards, unsafe_allow_html=True)

# Quality pill
qp_cls = {"green":"qp-green","orange":"qp-orange","red":"qp-red"}.get(color_key,"qp-green")
st.markdown(
    f'<div class="quality-pill {qp_cls}">'
    f'<span class="qp-score">{score}'
    f'<span style="font-size:1.05rem;font-weight:600;">/100</span></span>'
    f'<div><div class="qp-label">🏅 {label} — Data Quality Score</div>'
    f'<div class="qp-detail">Completeness {quality["completeness"]}%'
    f' &nbsp;·&nbsp; Uniqueness {quality["uniqueness"]}%'
    f' &nbsp;·&nbsp; Consistency {quality["consistency"]}%</div></div></div>',
    unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_viz, tab_ai, tab_story = st.tabs(
    ["📋  Overview", "📈  Visualizations", "🤖  AI Insights", "📖  Data Story"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── Smart Recommendations ─────────────────────────────────────────────────
    recs = _smart_recommendations(profile, quality, anomalies)
    with st.container(border=True):
        _chart_card("🎯", "Smart Recommendations", f"{len(recs)} action item(s) found")
        rec_html = '<div class="chart-card-body"><div class="rec-grid">'
        for r in recs:
            rec_html += (f'<div class="rec-card" style="--rc:{r["color"]};--rbg:{r["bg"]};">'
                         f'<div class="rec-icon">{r["icon"]}</div><div>'
                         f'<div class="rec-title">{r["title"]}</div>'
                         f'<div class="rec-body">{r["body"]}</div></div></div>')
        rec_html += '</div></div>'
        st.markdown(rec_html, unsafe_allow_html=True)

    # ── Data Quality ──────────────────────────────────────────────────────────
    with st.container(border=True):
        _chart_card("🏅", "Data Quality", f"Overall score: {score}/100 — {label}")
        qcol_l, qcol_r = st.columns([1, 1])
        with qcol_l:
            st.plotly_chart(plot_quality_radar(quality), width='stretch')
        with qcol_r:
            quality_df = pd.DataFrame({
                "Metric": ["Overall Score","Completeness","Uniqueness",
                           "Consistency","Missing Cells","Duplicate Rows"],
                "Value": [f"{quality['overall_score']} / 100  ({label})",
                          f"{quality['completeness']}%", f"{quality['uniqueness']}%",
                          f"{quality['consistency']}%", f"{quality['missing_cells']:,}",
                          f"{quality['duplicate_rows']:,}"],
            })
            st.dataframe(quality_df, use_container_width=True, hide_index=True, height=245)

    # ── Dataset Preview ───────────────────────────────────────────────────────
    with st.container(border=True):
        _chart_card("🔍", "Dataset Preview", f"Showing {preview_rows} rows")
        st.dataframe(safe_sample_df(df, preview_rows), use_container_width=True)

    # ── Column Summary + Descriptive Stats ────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        with st.container(border=True):
            _chart_card("🗂", "Column Summary", f"{profile['columns']} columns")
            st.dataframe(column_type_summary(df), use_container_width=True, hide_index=True)
    with col_r:
        if profile["descriptive_stats"] is not None:
            with st.container(border=True):
                _chart_card("📐", "Descriptive Statistics", f"{profile['numeric_columns']} numeric features")
                st.dataframe(profile["descriptive_stats"], use_container_width=True)

    # ── Strong Correlations ───────────────────────────────────────────────────
    if profile.get("high_correlations"):
        with st.container(border=True):
            _chart_card("🔗", "Strong Correlations", "|r| ≥ 0.7")
            corr_df = pd.DataFrame(profile["high_correlations"])
            corr_df.columns = ["Feature A", "Feature B", "Pearson r"]
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

    # ── Detected Outliers ─────────────────────────────────────────────────────
    with st.container(border=True):
        _chart_card("⚠️", "Detected Outliers", "IQR method")
        if anomalies:
            st.dataframe(pd.DataFrame([
                {"Column": col, "Outlier Count": i["count"], "Outlier %": f"{i['percent']}%",
                 "Valid Range": f"[{i['lower_bound']}, {i['upper_bound']}]",
                 "Min": i["min_outlier"], "Max": i["max_outlier"]}
                for col, i in anomalies.items()
            ]), use_container_width=True, hide_index=True)
        else:
            st.success("No significant outliers detected.", icon="✅")

    # ── Feature Importance ────────────────────────────────────────────────────
    if not importance.empty:
        with st.container(border=True):
            _chart_card("📊", "Feature Importance", "Variance-based ranking")
            st.dataframe(importance, use_container_width=True, hide_index=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    with st.container(border=True):
        _chart_card("📥", "Downloads", "Export processed data & full report")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇  Download Dataset (CSV)", data=df_to_csv_bytes(df),
                               file_name=f"processed_{uploaded_file.name}", mime="text/csv",
                               use_container_width=True)
        with dl2:
            cached_insights = st.session_state.get("insights", "— AI insights not yet generated —")
            report_text = build_text_report(profile, quality, cached_insights, uploaded_file.name)
            st.download_button("⬇  Download Analysis Report (.txt)", data=report_text.encode("utf-8"),
                               file_name=f"report_{uploaded_file.name.replace('.csv','')}.txt",
                               mime="text/plain", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — VISUALIZATIONS  (Dashboard-style layout)
# ════════════════════════════════════════════════════════════════════════════════
with tab_viz:

    # ── Row: Missing values (only if relevant) ─────────────────────────────
    fig_miss = plot_missing_values(df)
    if fig_miss:
        st.markdown('<p class="viz-section-label">Data Completeness</p>', unsafe_allow_html=True)
        with st.container(border=True):
            _chart_card("🕳", "Missing Values by Column", "% of null cells per feature")
            st.plotly_chart(fig_miss, width='stretch')

    # ── Row: Distributions + Heatmap side by side ──────────────────────────
    fig_hist = plot_histograms(df)
    fig_corr = plot_correlation_heatmap(df)

    st.markdown('<p class="viz-section-label">Distributions & Correlations</p>', unsafe_allow_html=True)

    if fig_hist and fig_corr:
        c_left, c_right = st.columns([3, 2])
        with c_left:
            with st.container(border=True):
                _chart_card("📊", "Feature Distributions", "Histogram per numeric column")
                st.plotly_chart(fig_hist, width='stretch')
        with c_right:
            with st.container(border=True):
                _chart_card("🔥", "Correlation Heatmap", "Pearson r between numeric features")
                st.plotly_chart(fig_corr, width='stretch')
    elif fig_hist:
        with st.container(border=True):
            _chart_card("📊", "Feature Distributions", "Histogram per numeric column")
            st.plotly_chart(fig_hist, width='stretch')
    elif fig_corr:
        with st.container(border=True):
            _chart_card("🔥", "Correlation Heatmap", "Pearson r between numeric features")
            st.plotly_chart(fig_corr, width='stretch')

    # ── Row: Correlation Network ───────────────────────────────────────────
    fig_net = plot_correlation_network(df)
    if fig_net:
        st.markdown('<p class="viz-section-label">Network Analysis</p>', unsafe_allow_html=True)
        with st.container(border=True):
            _chart_card("🕸", "Correlation Network", "Nodes = features · Edge width ∝ |r| · Blue = positive · Rose = negative · threshold |r| ≥ 0.4")
            st.plotly_chart(fig_net, width='stretch')

    # ── Row: Boxplots + Feature Importance ────────────────────────────────
    fig_box = plot_boxplots(df)
    fig_imp = plot_feature_importance(importance)

    st.markdown('<p class="viz-section-label">Outliers & Feature Importance</p>', unsafe_allow_html=True)

    if fig_box and fig_imp:
        c_left, c_right = st.columns([3, 2])
        with c_left:
            with st.container(border=True):
                _chart_card("📦", "Boxplots", "IQR outlier detection across numeric features")
                st.plotly_chart(fig_box, width='stretch')
        with c_right:
            with st.container(border=True):
                _chart_card("🎯", "Feature Importance", importance.iloc[0]["Method"] if not importance.empty else "")
                st.plotly_chart(fig_imp, width='stretch')
    elif fig_box:
        with st.container(border=True):
            _chart_card("📦", "Boxplots", "IQR outlier detection")
            st.plotly_chart(fig_box, width='stretch')
    elif fig_imp:
        with st.container(border=True):
            _chart_card("🎯", "Feature Importance", "")
            st.plotly_chart(fig_imp, width='stretch')

    # ── Row: Categorical distributions (2-up grid) ────────────────────────
    cat_figs = plot_categorical_distributions(df)
    if cat_figs:
        st.markdown('<p class="viz-section-label">Categorical Distributions</p>', unsafe_allow_html=True)
        for i in range(0, len(cat_figs), 2):
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.plotly_chart(cat_figs[i], width='stretch')
            if i + 1 < len(cat_figs):
                with c2:
                    with st.container(border=True):
                        st.plotly_chart(cat_figs[i + 1], width='stretch')


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_ai:
    with st.container(border=True):
        _chart_card("🤖", "Claude-Powered Analysis", "AI-generated findings from your dataset")
        st.markdown(
            '<p style="color:#475569;font-size:0.88rem;margin:10px 4px 14px;">'
            'Claude reads your full dataset profile and returns structured findings: '
            'key insights, trends, anomalies, business questions, and next steps.</p>',
            unsafe_allow_html=True,
        )

        if not api_key:
            st.markdown("""<div class="empty-state"><div class="es-icon">🔑</div>
                <div class="es-title">API Key Required</div>
                <div class="es-body">Enter your <strong>Anthropic API key</strong> in the sidebar.</div>
            </div>""", unsafe_allow_html=True)
        else:
            if st.button("✨  Generate AI Insights", type="primary", key="btn_insights"):
                with st.spinner("Analysing your dataset with Llama 3 …"):
                    try:
                        client   = Groq(api_key=api_key)
                        insights = generate_insights(client, profile, quality, anomalies)
                        st.session_state["insights"]      = insights
                        st.session_state["insights_file"] = uploaded_file.name
                    except Exception as exc:
                        err = str(exc)
                        if "invalid_api_key" in err or "401" in err:
                            st.error("Invalid Groq API key — check your key and retry.", icon="🚫")
                        elif "rate_limit" in err or "429" in err:
                            st.error("Rate limit reached. Wait a moment then retry.", icon="⏳")
                        else:
                            st.error(f"Unexpected error: {exc}", icon="❌")

            if ("insights" in st.session_state
                    and st.session_state.get("insights_file") == uploaded_file.name):
                st.markdown(f'<div class="insight-card">{st.session_state["insights"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown("""<div class="empty-state"><div class="es-icon">🤖</div>
                    <div class="es-title">Ready to Analyse</div>
                    <div class="es-body">Click <strong>Generate AI Insights</strong> above.</div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA STORY
# ════════════════════════════════════════════════════════════════════════════════
with tab_story:
    with st.container(border=True):
        _chart_card("📖", "Your Data Story", "Plain-English narrative for any audience")
        st.markdown(
            '<p style="color:#475569;font-size:0.88rem;margin:10px 4px 14px;">'
            'A jargon-free, three-paragraph narrative — no bullet points, no statistics — '
            'written for any audience. Ideal for executive summaries and presentations.</p>',
            unsafe_allow_html=True,
        )

        if not enable_story:
            st.markdown("""<div class="empty-state"><div class="es-icon">⚙️</div>
                <div class="es-title">Feature Disabled</div>
                <div class="es-body">Enable <strong>Data Story</strong> in the sidebar options.</div>
            </div>""", unsafe_allow_html=True)
        elif not api_key:
            st.markdown("""<div class="empty-state"><div class="es-icon">🔑</div>
                <div class="es-title">API Key Required</div>
                <div class="es-body">Enter your <strong>Anthropic API key</strong> in the sidebar.</div>
            </div>""", unsafe_allow_html=True)
        elif ("insights" not in st.session_state
              or st.session_state.get("insights_file") != uploaded_file.name):
            st.markdown("""<div class="empty-state"><div class="es-icon">🤖</div>
                <div class="es-title">Generate Insights First</div>
                <div class="es-body">Go to <strong>AI Insights</strong>, generate insights, then return here.</div>
            </div>""", unsafe_allow_html=True)
        else:
            if st.button("✨  Generate Data Story", type="primary", key="btn_story"):
                with st.spinner("Writing your data story with Llama 3 …"):
                    try:
                        client = Groq(api_key=api_key)
                        story  = generate_data_story(client, profile, st.session_state["insights"])
                        st.session_state["data_story"] = story
                        st.session_state["story_file"] = uploaded_file.name
                    except Exception as exc:
                        err = str(exc)
                        if "invalid_api_key" in err or "401" in err:
                            st.error("Invalid Groq API key — check your key and retry.", icon="🚫")
                        elif "rate_limit" in err or "429" in err:
                            st.error("Rate limit hit. Wait a moment then retry.", icon="⏳")
                        else:
                            st.error(f"Unexpected error: {exc}", icon="❌")

            if ("data_story" in st.session_state
                    and st.session_state.get("story_file") == uploaded_file.name):
                story_html = st.session_state["data_story"].replace("\n", "<br>")
                st.markdown(f'<div class="story-card">{story_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown("""<div class="empty-state"><div class="es-icon">📖</div>
                    <div class="es-title">Story Not Yet Generated</div>
                    <div class="es-body">Click <strong>Generate Data Story</strong> above.</div>
                </div>""", unsafe_allow_html=True)
