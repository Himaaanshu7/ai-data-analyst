"""
ai_insights.py - Groq API integration for AI-generated insights and data storytelling.

Two public functions:
    generate_insights()    → detailed technical analysis (markdown)
    generate_data_story()  → plain-English narrative for non-technical readers
"""

from groq import Groq

_MODEL               = "llama-3.3-70b-versatile"
_MAX_TOKENS_INSIGHTS = 2048
_MAX_TOKENS_STORY    = 1024


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_insights_prompt(profile: dict, quality: dict, anomalies: dict) -> str:
    lines = []

    lines += [
        "You are an expert data scientist and analyst.",
        "Carefully read the dataset summary below and provide a structured analysis.",
        "",
        "## Dataset Overview",
        f"- **Shape:** {profile['rows']:,} rows × {profile['columns']} columns",
        f"- **Memory:** {profile['memory_usage_mb']} MB",
    ]

    if profile["numeric_col_names"]:
        cols_preview = profile["numeric_col_names"][:12]
        suffix = (f" … (+{len(profile['numeric_col_names']) - 12} more)"
                  if len(profile["numeric_col_names"]) > 12 else "")
        lines.append(f"- **Numeric columns ({profile['numeric_columns']}):** "
                     f"{', '.join(cols_preview)}{suffix}")
    if profile["categorical_col_names"]:
        cols_preview = profile["categorical_col_names"][:8]
        suffix = (f" … (+{len(profile['categorical_col_names']) - 8} more)"
                  if len(profile["categorical_col_names"]) > 8 else "")
        lines.append(f"- **Categorical columns ({profile['categorical_columns']}):** "
                     f"{', '.join(cols_preview)}{suffix}")
    if profile["datetime_col_names"]:
        lines.append(f"- **Datetime columns ({profile['datetime_columns']}):** "
                     f"{', '.join(profile['datetime_col_names'])}")

    lines += [
        f"- **Total missing values:** {profile['total_missing']:,}",
        f"- **Duplicate rows:** {profile['duplicate_rows']:,}",
        "",
    ]

    lines += [
        "## Data Quality Score",
        f"- **Overall:** {quality['overall_score']}/100",
        f"- Completeness: {quality['completeness']}%",
        f"- Uniqueness:   {quality['uniqueness']}%",
        f"- Consistency:  {quality['consistency']}%",
        "",
    ]

    stats = profile.get("descriptive_stats")
    if stats is not None:
        lines += ["## Descriptive Statistics (Numeric Columns)", stats.to_string(), ""]

    high_corr = profile.get("high_correlations", [])
    if high_corr:
        lines.append("## Strong Correlations (|r| ≥ 0.7)")
        for hc in high_corr[:10]:
            lines.append(f"- **{hc['col1']}** ↔ **{hc['col2']}**: r = {hc['correlation']}")
        lines.append("")

    cat_counts = profile.get("categorical_value_counts", {})
    if cat_counts:
        lines.append("## Categorical Column Distributions (Top Values)")
        for col, counts in list(cat_counts.items())[:6]:
            top5 = list(counts.items())[:5]
            vals = ", ".join(f"{k} ({v})" for k, v in top5)
            lines.append(f"- **{col}:** {vals}")
        lines.append("")

    if anomalies:
        lines.append("## Detected Outliers (IQR Method)")
        for col, info in list(anomalies.items())[:10]:
            lines.append(f"- **{col}:** {info['count']} outliers ({info['percent']}%) "
                         f"— valid range [{info['lower_bound']}, {info['upper_bound']}]")
        lines.append("")

    cols_with_missing = {c: i for c, i in profile["missing_per_column"].items() if i["count"] > 0}
    if cols_with_missing:
        lines.append("## Missing Values by Column")
        for col, info in list(cols_with_missing.items())[:10]:
            lines.append(f"- **{col}:** {info['count']:,} missing ({info['percent']}%)")
        lines.append("")

    lines += [
        "---",
        "Based on the above summary, provide a **structured analysis** with:",
        "",
        "### 1. Key Insights",
        "3–5 of the most important patterns, distributions, or relationships.",
        "",
        "### 2. Potential Trends",
        "2–3 trends or patterns that are visible or strongly implied.",
        "",
        "### 3. Anomalies & Data Quality Issues",
        "Flag outliers, skewed distributions, high missingness, or quality concerns.",
        "",
        "### 4. Suggested Business Questions",
        "3–4 questions a business analyst or stakeholder should explore further.",
        "",
        "### 5. Recommended Next Steps",
        "What transformations, feature engineering, or analyses should be done next?",
        "",
        "**Important:** Reference actual column names and values. Use markdown formatting. "
        "Be specific — avoid generic advice that could apply to any dataset.",
    ]

    return "\n".join(lines)


def _build_story_prompt(profile: dict, insights: str) -> str:
    lines = [
        "You are a professional data storyteller writing for a non-technical business audience.",
        "",
        "Dataset facts:",
        f"- {profile['rows']:,} records, {profile['columns']} features",
        f"- Numeric features: {profile['numeric_columns']}",
        f"- Categorical features: {profile['categorical_columns']}",
        "",
        "The technical analysis produced these findings:",
        "---",
        insights[:2000],
        "---",
        "",
        "Write a **3-paragraph data story** that:",
        "1. Opens with an engaging summary of what this dataset represents and its scope.",
        "2. Explains the 2–3 most interesting patterns in plain English (no statistical jargon).",
        "3. Closes with 2–3 clear, actionable recommendations.",
        "",
        "Rules: ≤300 words total, no bullet points, no markdown headers, "
        "write in a warm and confident business tone.",
    ]
    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_insights(client: Groq, profile: dict, quality: dict, anomalies: dict) -> str:
    prompt = _build_insights_prompt(profile, quality, anomalies)
    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS_INSIGHTS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_data_story(client: Groq, profile: dict, insights: str) -> str:
    prompt = _build_story_prompt(profile, insights)
    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS_STORY,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
