"""
data_analysis.py - Dataset profiling, anomaly detection, and feature importance.
"""

import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Compute a comprehensive profile of the dataset.

    Returns a dict with:
        rows, columns, numeric/categorical/datetime column counts and names,
        missing value info, duplicate count, descriptive stats, correlations,
        high-correlation pairs, categorical value counts, memory usage.
    """
    profile = {}

    # ── Shape ──────────────────────────────────────────────────────────────────
    profile["rows"] = len(df)
    profile["columns"] = len(df.columns)

    # ── Column type breakdown ──────────────────────────────────────────────────
    numeric_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols   = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    profile["numeric_columns"]      = len(numeric_cols)
    profile["categorical_columns"]  = len(categorical_cols)
    profile["datetime_columns"]     = len(datetime_cols)
    profile["numeric_col_names"]    = numeric_cols
    profile["categorical_col_names"] = categorical_cols
    profile["datetime_col_names"]   = datetime_cols
    profile["all_columns"]          = df.columns.tolist()

    # ── Missing values ─────────────────────────────────────────────────────────
    missing = df.isnull().sum()
    profile["total_missing"] = int(missing.sum())
    profile["missing_per_column"] = {
        col: {
            "count":   int(missing[col]),
            "percent": round(missing[col] / len(df) * 100, 2) if len(df) > 0 else 0,
        }
        for col in df.columns
    }

    # ── Duplicates ─────────────────────────────────────────────────────────────
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # ── Descriptive statistics ─────────────────────────────────────────────────
    if numeric_cols:
        profile["descriptive_stats"] = df[numeric_cols].describe().round(4)
    else:
        profile["descriptive_stats"] = None

    # ── Correlations ───────────────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(4)
        profile["correlation_matrix"] = corr_matrix

        high_corr = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= 0.7:
                    high_corr.append({
                        "col1":        cols[i],
                        "col2":        cols[j],
                        "correlation": round(float(val), 4),
                    })
        high_corr.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        profile["high_correlations"] = high_corr
    else:
        profile["correlation_matrix"] = None
        profile["high_correlations"]  = []

    # ── Categorical value counts (top 10 per column) ───────────────────────────
    cat_value_counts = {}
    for col in categorical_cols:
        cat_value_counts[col] = df[col].value_counts().head(10).to_dict()
    profile["categorical_value_counts"] = cat_value_counts

    # ── Memory usage ───────────────────────────────────────────────────────────
    profile["memory_usage_mb"] = round(
        df.memory_usage(deep=True).sum() / 1024 ** 2, 3
    )

    return profile


def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Detect outliers in numeric columns using the IQR (Tukey) method.

    A value is an outlier if it falls below Q1 - 1.5×IQR or above Q3 + 1.5×IQR.
    Returns a dict keyed by column name; only columns with outliers are included.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    anomalies = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:          # need enough data for IQR to be meaningful
            continue

        q1  = series.quantile(0.25)
        q3  = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:                 # constant column — skip
            continue

        lower    = q1 - 1.5 * iqr
        upper    = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]

        if len(outliers) > 0:
            anomalies[col] = {
                "count":       int(len(outliers)),
                "percent":     round(len(outliers) / len(series) * 100, 2),
                "lower_bound": round(float(lower),          4),
                "upper_bound": round(float(upper),          4),
                "min_outlier": round(float(outliers.min()), 4),
                "max_outlier": round(float(outliers.max()), 4),
            }

    return anomalies


def compute_feature_importance(
    df: pd.DataFrame, target_col: str = None
) -> pd.DataFrame:
    """
    Estimate feature importance for numeric columns.

    - If *target_col* is provided and is numeric: uses |Pearson correlation|
      with the target column.
    - Otherwise: uses variance share (each column's variance as a % of total
      variance across all numeric columns).

    Returns a DataFrame with columns [Feature, Importance, Method].
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame(columns=["Feature", "Importance", "Method"])

    if target_col and target_col in numeric_cols:
        corr = (
            df[numeric_cols]
            .corr()[target_col]
            .drop(target_col)
            .abs()
            .sort_values(ascending=False)
        )
        result = pd.DataFrame({
            "Feature":    corr.index,
            "Importance": corr.values.round(4),
            "Method":     "Pearson |r| with target",
        })
    else:
        variances = df[numeric_cols].var()
        total_var = variances.sum()
        if total_var > 0:
            importance = (variances / total_var).sort_values(ascending=False)
        else:
            importance = variances.sort_values(ascending=False)

        result = pd.DataFrame({
            "Feature":    importance.index,
            "Importance": importance.values.round(4),
            "Method":     "Variance share",
        })

    return result.reset_index(drop=True)
