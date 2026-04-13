"""
visualization.py - Plotly chart generators for DataLens AI.

Design principles:
  - Every chart column/category gets a unique palette colour
  - Generous margins so axis labels are never clipped
  - Fonts ≥ 12 px everywhere
  - Semi-transparent fills so structure (IQR, bins) is always readable
"""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Brand palette — 10 distinct, high-contrast colours ───────────────────────
_PALETTE = [
    "#4f46e5",  # indigo
    "#0891b2",  # cyan-600
    "#059669",  # emerald
    "#d97706",  # amber
    "#dc2626",  # red
    "#7c3aed",  # violet
    "#0284c7",  # sky
    "#16a34a",  # green
    "#db2777",  # pink
    "#9333ea",  # purple
]

_FONT = dict(family="Inter, system-ui, sans-serif")


def _rgba(hex_color: str, alpha: float = 0.22) -> str:
    """Convert a hex colour to rgba string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(**kwargs) -> dict:
    """Shared layout defaults for every chart.
    NOTE: Do NOT set title_font here — Plotly renders a blank title as 'undefined'.
    Each chart that needs a title sets it explicitly via title=dict(text=...).
    """
    return dict(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=_FONT,
        hoverlabel=dict(
            font=_FONT,
            bgcolor="#1e293b",
            font_color="#f8fafc",
            bordercolor="#334155",
        ),
        **kwargs,
    )


def _label_margin(labels, char_px: int = 9, minimum: int = 130, maximum: int = 280) -> int:
    """Compute left/bottom margin so the longest label is never clipped."""
    longest = max((len(str(lbl)) for lbl in labels), default=10)
    return min(max(longest * char_px, minimum), maximum)


def _truncate(text: str, n: int = 28) -> str:
    return str(text)[:n] + "…" if len(str(text)) > n else str(text)


# ── Histograms ─────────────────────────────────────────────────────────────────

def plot_histograms(df: pd.DataFrame, max_cols: int = 12) -> go.Figure | None:
    """
    Grid of histograms — each column gets its own palette colour with a white
    bar-outline so individual bins are always legible.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if not numeric_cols:
        return None

    n            = len(numeric_cols)
    cols_per_row = min(3, n)
    rows         = math.ceil(n / cols_per_row)

    fig = make_subplots(
        rows=rows,
        cols=cols_per_row,
        subplot_titles=numeric_cols,
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    for i, col in enumerate(numeric_cols):
        r       = i // cols_per_row + 1
        c       = i  % cols_per_row + 1
        colour  = _PALETTE[i % len(_PALETTE)]
        data    = df[col].dropna()

        fig.add_trace(
            go.Histogram(
                x=data,
                name=col,
                nbinsx=20,
                marker=dict(
                    color=_rgba(colour, 0.75),
                    line=dict(color=colour, width=1.2),
                ),
                showlegend=False,
                hovertemplate=f"<b>{col}</b><br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ),
            row=r, col=c,
        )

    # Uniform axis styling for every subplot
    fig.update_xaxes(
        tickfont=dict(size=11, color="#475569"),
        title_font=dict(size=11, color="#64748b"),
        gridcolor="#e2e8f0",
        showgrid=True,
    )
    fig.update_yaxes(
        tickfont=dict(size=11, color="#475569"),
        gridcolor="#e2e8f0",
        showgrid=True,
    )

    # Bold subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#1e293b", **_FONT)

    fig.update_layout(
        **_base_layout(),
        height=300 * rows + 60,
        margin=dict(t=40, b=50, l=60, r=20),
    )
    return fig


# ── Correlation heatmap ────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """Annotated correlation heatmap with readable axis labels."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr().round(2)
    lm   = _label_margin(numeric_cols, char_px=8, minimum=80, maximum=200)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont=dict(size=11, color="#1e293b"),
        colorbar=dict(
            title=dict(text="r", font=dict(size=12, color="#475569")),
            tickfont=dict(size=11, color="#475569"),
            thickness=14, len=0.85,
        ),
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(),
        height=max(440, len(numeric_cols) * 52 + 100),
        xaxis=dict(
            tickangle=-40,
            tickfont=dict(size=12, color="#1e293b"),
            title_font=dict(size=12),
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#1e293b"),
            autorange="reversed",
        ),
        margin=dict(t=40, b=lm, l=lm, r=20),
    )
    return fig


# ── Boxplots ───────────────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, max_cols: int = 15) -> go.Figure | None:
    """
    Side-by-side boxplots.  Each column uses a semi-transparent fill +
    solid-colour border so IQR structure is always visible.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if not numeric_cols:
        return None

    fig = go.Figure()
    for i, col in enumerate(numeric_cols):
        colour = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            fillcolor=_rgba(colour, 0.25),
            line=dict(color=colour, width=2),
            marker=dict(
                color=colour,
                size=4,
                opacity=0.65,
                symbol="circle",
            ),
            boxpoints="outliers",
            jitter=0.3,
            hovertemplate=(
                f"<b>{col}</b><br>"
                "Q1: %{q1:.2f}<br>Median: %{median:.2f}<br>"
                "Q3: %{q3:.2f}<extra></extra>"
            ),
        ))

    lm = _label_margin(numeric_cols, char_px=8, minimum=60, maximum=160)

    fig.update_layout(
        **_base_layout(),
        height=520,
        showlegend=False,
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=12, color="#1e293b"),
            title_font=dict(size=12),
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#475569"),
            gridcolor="#e2e8f0",
            title="Value",
            title_font=dict(size=12, color="#64748b"),
        ),
        margin=dict(t=30, b=lm, l=70, r=20),
    )
    return fig


# ── Categorical bar charts ─────────────────────────────────────────────────────

def plot_categorical_distributions(
    df: pd.DataFrame, max_cols: int = 8
) -> list[go.Figure]:
    """
    One horizontal bar chart per categorical column (top 15 values).
    Uses a single palette colour per chart for maximum readability;
    bar width and left margin adapt to the longest label.
    """
    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()[:max_cols]

    figs = []
    for idx, col in enumerate(cat_cols):
        colour  = _PALETTE[idx % len(_PALETTE)]
        counts  = df[col].value_counts().head(15)
        labels  = [_truncate(v) for v in counts.index]
        lm      = _label_margin(labels, char_px=8, minimum=140, maximum=300)

        fig = go.Figure(go.Bar(
            x=counts.values,
            y=labels,
            orientation="h",
            marker=dict(
                color=_rgba(colour, 0.80),
                line=dict(color=colour, width=1.5),
            ),
            text=[f" {v:,}" for v in counts.values],
            textposition="outside",
            textfont=dict(size=11, color="#1e293b"),
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        ))

        fig.update_layout(
            **_base_layout(),
            title=dict(
                text=f"<b>{col}</b> — top {len(counts)} values",
                font=dict(size=14, color="#0f172a", **_FONT),
                x=0, xanchor="left",
            ),
            height=max(360, len(counts) * 36 + 100),
            xaxis=dict(
                title="Count",
                title_font=dict(size=12, color="#64748b"),
                tickfont=dict(size=11, color="#475569"),
                gridcolor="#e2e8f0",
                range=[0, counts.max() * 1.22],
            ),
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(size=12, color="#1e293b"),
                tickmode="array",
                tickvals=list(range(len(labels))),
                ticktext=labels,
            ),
            margin=dict(t=50, b=30, l=lm, r=70),
            showlegend=False,
        )
        figs.append(fig)

    return figs


# ── Missing values bar ─────────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame) -> go.Figure | None:
    """Horizontal bar chart of missing-value % per column."""
    missing = df.isnull().mean().mul(100).round(2)
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        return None

    lm = _label_margin(missing.index, char_px=9, minimum=130, maximum=280)

    fig = go.Figure(go.Bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        marker=dict(
            color=[_rgba("#f43f5e", 0.20 + 0.65 * (v / 100)) for v in missing.values],
            line=dict(color="#f43f5e", width=1.5),
        ),
        text=[f" {v:.1f}%" for v in missing.values],
        textposition="outside",
        textfont=dict(size=11, color="#1e293b"),
        hovertemplate="<b>%{y}</b><br>Missing: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(),
        height=max(320, len(missing) * 40 + 100),
        xaxis=dict(
            title="Missing %",
            title_font=dict(size=12, color="#64748b"),
            tickfont=dict(size=11, color="#475569"),
            range=[0, min(missing.max() * 1.3, 110)],
            gridcolor="#e2e8f0",
        ),
        yaxis=dict(tickfont=dict(size=12, color="#1e293b")),
        margin=dict(t=30, b=40, l=lm, r=70),
        showlegend=False,
    )
    return fig


# ── Feature importance ─────────────────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure | None:
    """Horizontal bar chart with a colour gradient that spans the full range."""
    if importance_df is None or importance_df.empty:
        return None

    top    = importance_df.head(15).copy().sort_values("Importance")
    method = top["Method"].iloc[0] if "Method" in top.columns else ""
    lm     = _label_margin(top["Feature"], char_px=9, minimum=130, maximum=280)

    # Normalise for colour mapping
    vals    = top["Importance"].values
    normed  = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    colours = [
        f"rgba({int(79 + 60*n)},{int(70 + 30*n)},{int(229 - 60*n)},0.85)"
        for n in normed
    ]

    fig = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker=dict(color=colours, line=dict(color="#4f46e5", width=1)),
        text=[f" {v:.3f}" for v in top["Importance"]],
        textposition="outside",
        textfont=dict(size=11, color="#1e293b"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(),
        title=dict(
            text=f"<b>Feature Importance</b> — {method}",
            font=dict(size=14, color="#0f172a", **_FONT),
            x=0, xanchor="left",
        ),
        height=max(320, len(top) * 38 + 100),
        xaxis=dict(
            title="Importance Score",
            title_font=dict(size=12, color="#64748b"),
            tickfont=dict(size=11, color="#475569"),
            range=[0, top["Importance"].max() * 1.22],
            gridcolor="#e2e8f0",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color="#1e293b"),
        ),
        margin=dict(t=50, b=30, l=lm, r=70),
        showlegend=False,
    )
    return fig


# ── Quality radar ──────────────────────────────────────────────────────────────

def plot_quality_radar(quality: dict) -> go.Figure:
    """Spider chart showing the three quality dimensions."""
    cats   = ["Completeness", "Uniqueness", "Consistency"]
    vals   = [quality["completeness"], quality["uniqueness"], quality["consistency"]]
    closed = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=closed, theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(79,70,229,0.15)",
        line=dict(color="#4f46e5", width=2.5),
        hovertemplate="%{theta}: <b>%{r:.1f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[100] * (len(cats) + 1), theta=cats + [cats[0]],
        line=dict(color="rgba(99,102,241,0.12)", width=1, dash="dot"),
        showlegend=False, hoverinfo="none",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="#f8fafc",
            radialaxis=dict(
                visible=True, range=[0, 100], ticksuffix="%",
                tickfont=dict(size=10, color="#64748b"),
                gridcolor="rgba(99,102,241,0.15)",
                linecolor="rgba(99,102,241,0.15)",
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color="#1e293b", **_FONT),
                gridcolor="rgba(99,102,241,0.1)",
            ),
        ),
        showlegend=False,
        height=300,
        paper_bgcolor="#ffffff",
        font=_FONT,
        margin=dict(t=30, b=30, l=50, r=50),
    )
    return fig


# ── Correlation network ────────────────────────────────────────────────────────

def plot_correlation_network(df: pd.DataFrame, threshold: float = 0.4) -> go.Figure | None:
    """
    Circular force-layout network.  Node size = degree (connection count).
    Blue edges = positive r, rose = negative r.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        return None

    corr = df[numeric_cols].corr()
    n    = len(numeric_cols)

    angles  = [2 * math.pi * i / n for i in range(n)]
    xs      = [round(math.cos(a), 6) for a in angles]
    ys      = [round(math.sin(a), 6) for a in angles]

    edge_traces = []
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr.iloc[i, j])
            if abs(r) < threshold:
                continue
            colour  = "#4f46e5" if r > 0 else "#f43f5e"
            opacity = 0.3 + 0.6 * abs(r)
            edge_traces.append(go.Scatter(
                x=[xs[i], xs[j], None], y=[ys[i], ys[j], None],
                mode="lines",
                line=dict(color=colour, width=1 + abs(r) * 5),
                opacity=opacity, showlegend=False, hoverinfo="none",
            ))

    degrees    = [sum(1 for j in range(n) if j != i and abs(corr.iloc[i, j]) >= threshold)
                  for i in range(n)]
    node_sizes = [20 + d * 6 for d in degrees]

    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=numeric_cols,
        textposition="top center",
        textfont=dict(size=12, color="#0f172a", **_FONT),
        marker=dict(
            size=node_sizes, color="#4f46e5",
            line=dict(color="white", width=2.5),
            opacity=0.9,
        ),
        hovertemplate=[
            f"<b>{col}</b><br>Connections: {degrees[i]}<extra></extra>"
            for i, col in enumerate(numeric_cols)
        ],
        showlegend=False,
    )

    annotations = [
        dict(x=1.01, y=0.58, xref="paper", yref="paper", showarrow=False,
             text="<b style='color:#4f46e5'>─</b> Positive r",
             font=dict(size=12, color="#4f46e5")),
        dict(x=1.01, y=0.48, xref="paper", yref="paper", showarrow=False,
             text="<b style='color:#f43f5e'>─</b> Negative r",
             font=dict(size=12, color="#f43f5e")),
        dict(x=1.01, y=0.38, xref="paper", yref="paper", showarrow=False,
             text=f"threshold |r| ≥ {threshold}",
             font=dict(size=10, color="#64748b")),
    ]

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        **_base_layout(),
        height=540,
        xaxis=dict(visible=False, range=[-1.4, 1.4]),
        yaxis=dict(visible=False, range=[-1.4, 1.4]),
        margin=dict(t=30, b=30, l=30, r=130),
        annotations=annotations,
    )
    return fig
