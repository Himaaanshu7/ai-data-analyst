"""
utils.py - Utility functions for report generation, formatting, and helpers.
"""

import io
import json
import pandas as pd
import numpy as np
from datetime import datetime


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number for display, handling NaN and infinity."""
    if pd.isna(value) or np.isinf(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:,.{decimals}f}"


def compute_dataset_quality_score(df: pd.DataFrame) -> dict:
    """
    Compute a dataset quality score (0-100) based on:
    - Completeness: % of non-missing values
    - Uniqueness: % of non-duplicate rows
    - Consistency: ratio of numeric cols with no infinite values
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 1

    duplicate_rows = df.duplicated().sum()
    uniqueness = 1 - (duplicate_rows / len(df)) if len(df) > 0 else 1

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_count = np.isinf(df[numeric_cols].values).sum()
        consistency = 1 - (inf_count / (len(numeric_cols) * len(df)))
        consistency = max(0, consistency)
    else:
        consistency = 1.0

    score = (completeness * 0.5 + uniqueness * 0.3 + consistency * 0.2) * 100

    return {
        "overall_score": round(score, 1),
        "completeness": round(completeness * 100, 1),
        "uniqueness": round(uniqueness * 100, 1),
        "consistency": round(consistency * 100, 1),
        "missing_cells": int(missing_cells),
        "duplicate_rows": int(duplicate_rows),
    }


def get_score_color(score: float) -> str:
    """Return a color string based on quality score."""
    if score >= 85:
        return "green"
    elif score >= 65:
        return "orange"
    else:
        return "red"


def get_score_label(score: float) -> str:
    """Return a label based on quality score."""
    if score >= 85:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 45:
        return "Fair"
    else:
        return "Poor"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def build_text_report(
    profile: dict,
    quality: dict,
    insights: str,
    dataset_name: str = "dataset",
) -> str:
    """
    Build a plain-text analysis report combining profiling stats and AI insights.
    """
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 70)
    lines.append(f"  AI-POWERED DATA ANALYSIS REPORT")
    lines.append(f"  Dataset : {dataset_name}")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 70)

    # Quality score
    lines.append("\n[DATASET QUALITY SCORE]")
    lines.append(f"  Overall Score  : {quality['overall_score']} / 100  ({get_score_label(quality['overall_score'])})")
    lines.append(f"  Completeness   : {quality['completeness']}%")
    lines.append(f"  Uniqueness     : {quality['uniqueness']}%")
    lines.append(f"  Consistency    : {quality['consistency']}%")
    lines.append(f"  Missing Cells  : {quality['missing_cells']}")
    lines.append(f"  Duplicate Rows : {quality['duplicate_rows']}")

    # Basic profile
    lines.append("\n[DATASET OVERVIEW]")
    lines.append(f"  Rows    : {profile.get('rows', 'N/A'):,}")
    lines.append(f"  Columns : {profile.get('columns', 'N/A')}")
    lines.append(f"  Numeric Columns      : {profile.get('numeric_columns', 0)}")
    lines.append(f"  Categorical Columns  : {profile.get('categorical_columns', 0)}")
    lines.append(f"  Datetime Columns     : {profile.get('datetime_columns', 0)}")
    lines.append(f"  Total Missing Values : {profile.get('total_missing', 0)}")
    lines.append(f"  Duplicate Rows       : {profile.get('duplicate_rows', 0)}")

    # Missing per column
    missing_info = profile.get("missing_per_column", {})
    if missing_info:
        lines.append("\n[MISSING VALUES PER COLUMN]")
        for col, info in missing_info.items():
            if info["count"] > 0:
                lines.append(f"  {col}: {info['count']} ({info['percent']:.1f}%)")

    # Descriptive stats
    stats = profile.get("descriptive_stats")
    if stats is not None:
        lines.append("\n[DESCRIPTIVE STATISTICS]")
        lines.append(stats.to_string())

    # AI Insights
    lines.append("\n" + "=" * 70)
    lines.append("  AI-GENERATED INSIGHTS (Claude)")
    lines.append("=" * 70)
    lines.append(insights if insights else "No insights generated.")

    lines.append("\n" + "=" * 70)
    lines.append("  End of Report")
    lines.append("=" * 70)

    return "\n".join(lines)


def safe_sample_df(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return a safe sample of the dataframe (head if rows < n)."""
    return df.head(min(n, len(df)))


def column_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame of column types and nullability."""
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
        unique_count = df[col].nunique()
        rows.append({
            "Column": col,
            "Type": dtype,
            "Non-Null Count": len(df) - null_count,
            "Null Count": null_count,
            "Null %": round(null_pct, 1),
            "Unique Values": unique_count,
        })
    return pd.DataFrame(rows)
