# DataLens AI — Project Documentation

**Version:** 1.0  
**Author:** Himanshu Mishra  
**Live App:** [datalens-ai.streamlit.app](https://datalens-ai.streamlit.app)  
**Repository:** [github.com/Himaaanshu7/ai-data-analyst](https://github.com/Himaaanshu7/ai-data-analyst)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Use Cases](#2-use-cases)
3. [Architecture](#3-architecture)
4. [Feature Walkthrough](#4-feature-walkthrough)
   - [Landing Page](#41-landing-page)
   - [Sidebar — Upload & Configuration](#42-sidebar--upload--configuration)
   - [Overview Tab](#43-overview-tab)
   - [Visualizations Tab](#44-visualizations-tab)
   - [AI Insights Tab](#45-ai-insights-tab)
   - [Data Story Tab](#46-data-story-tab)
5. [Technical Deep Dive](#5-technical-deep-dive)
6. [Data Flow](#6-data-flow)
7. [AI Integration](#7-ai-integration)
8. [Tech Stack](#8-tech-stack)

---

## 1. Project Overview

**DataLens AI** is a no-code, AI-powered data analysis assistant. Users upload any CSV file and the app instantly delivers a full exploratory data analysis (EDA) — complete with interactive visualisations, statistical profiling, anomaly detection, smart recommendations, and AI-generated insights using the **Groq API (Llama 3.3-70B)**.

The goal is to eliminate the need to write any Python or SQL for early-stage data exploration. A business analyst, data scientist, or student can upload raw data and walk away with meaningful findings in under 60 seconds.

> **Key differentiator:** AI insights run on Groq's free-tier Llama 3.3-70B — users pay nothing.

---

## 2. Use Cases

### Use Case 1 — Business Analyst: Sales Data Review

**Scenario:** A business analyst receives a monthly sales CSV with 10,000+ rows. They need to identify trends, outlier transactions, and missing records before presenting to management.

**How DataLens AI helps:**
1. Upload the CSV → instant data profile (rows, columns, missing %, duplicates)
2. Quality Score highlights completeness issues
3. Correlation heatmap reveals which variables drive revenue
4. AI Insights tab summarises the top 5 findings and recommends next steps
5. Data Story generates an executive-ready paragraph — no writing needed

---

### Use Case 2 — Data Scientist: Pre-Modelling EDA

**Scenario:** A data scientist needs to understand a new dataset before building a machine learning model.

**How DataLens AI helps:**
1. Feature Importance ranking shows which columns carry the most signal
2. Anomaly Detection (IQR method) flags outlier-heavy columns to handle before training
3. Correlation Network identifies multicollinear features that should be pruned
4. Smart Recommendations proactively suggests: imputation strategies, duplicate removal, zero-variance drops, potential classification targets (binary columns)

---

### Use Case 3 — Student: Academic Dataset Analysis

**Scenario:** A student needs to analyse a dataset for a university assignment and write a report.

**How DataLens AI helps:**
1. Descriptive Statistics table gives mean, std, min, max, quartiles
2. Histograms and boxplots visualise distributions for each variable
3. Data Story generates a clean 3-paragraph narrative ready to paste into a report
4. Download Analysis Report (.txt) exports everything to submit as an appendix

---

### Use Case 4 — Product Manager: User Behaviour Data

**Scenario:** A PM receives exported user event data and wants to understand engagement patterns without involving the engineering team.

**How DataLens AI helps:**
1. Categorical distributions show which features/events are most common
2. Missing value chart reveals which fields users skip during sign-up
3. AI Insights surfaces suggested business questions — e.g. "Which user segment has the highest churn risk?"
4. One-click report download for async sharing with stakeholders

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      app.py (UI Layer)                  │
│  Streamlit tabs · sidebar · session state · downloads   │
└────────┬────────────────┬────────────────┬──────────────┘
         │                │                │
         ▼                ▼                ▼
┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
│data_analysis│  │visualization │  │   ai_insights   │
│   .py       │  │    .py       │  │      .py        │
│             │  │              │  │                 │
│ profile_    │  │ plot_hist    │  │ generate_       │
│ dataset()   │  │ plot_heatmap │  │ insights()      │
│             │  │ plot_boxplot │  │                 │
│ detect_     │  │ plot_network │  │ generate_       │
│ anomalies() │  │ plot_radar   │  │ data_story()    │
│             │  │ ...          │  │                 │
│ compute_    │  └──────┬───────┘  └────────┬────────┘
│ feature_    │         │                   │
│ importance()│         ▼                   ▼
└──────┬──────┘   Plotly charts       Groq API
       │          (interactive)    (Llama 3.3-70B)
       ▼
┌─────────────┐
│   utils.py  │
│             │
│ quality_    │
│ score()     │
│             │
│ build_      │
│ report()    │
└─────────────┘
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, routing, layout, session state, downloads |
| `data_analysis.py` | Dataset profiling, anomaly detection, feature importance |
| `visualization.py` | All Plotly chart builders |
| `ai_insights.py` | Groq API prompts and calls |
| `utils.py` | Quality score, formatters, report builder |

---

## 4. Feature Walkthrough

### 4.1 Landing Page

When no file is uploaded, the app displays a hero section with:
- **App name and tagline** — bold white title on a dark gradient background
- **Feature chips** — quick-scan overview of what the app does
- **How it works** — 4-step numbered flow
- **What you get** — feature cards (Profiling, Dashboard, Recommendations, AI)
- **Sample datasets** — Titanic, Iris, Tips with context on why each is useful

> **Screenshot:** Hero banner with gradient background, chip tags, and step flow

---

### 4.2 Sidebar — Upload & Configuration

The sidebar contains three sections:

**1. API Key (auto-loaded)**  
The Groq API key is stored in `.streamlit/secrets.toml` and loaded automatically. The sidebar displays a masked version of the key — users never need to enter anything.

```
✅ Key loaded automatically
gsk_lwTc·········gX
```

**2. Upload Dataset**  
A custom-styled drag-and-drop zone accepts `.csv` files up to 50 MB. After upload, the filename and file size are shown below the dropzone.

**3. Options**  
- **Preview rows** — slider (5–50) controls how many rows appear in the Dataset Preview
- **Enable Data Story** — toggle to enable/disable the AI storytelling tab

---

### 4.3 Overview Tab

The Overview tab provides a complete statistical profile of the dataset, all displayed inside styled card containers with lavender headers.

#### Dataset Banner
Shows filename, row count, column count, and memory usage at a glance.

#### Metric Cards (6 cards)
| Card | Value |
|---|---|
| Rows | Total record count |
| Columns | Total feature count |
| Numeric | Count of numeric columns |
| Categorical | Count of text/category columns |
| Missing | Total missing cell count |
| Duplicates | Duplicate row count |

#### Quality Score Pill
A colour-coded pill (green/orange/red) showing the overall data quality score out of 100, with breakdown of Completeness, Uniqueness, and Consistency percentages.

**Quality Score Formula:**
```
Completeness  = (1 - missing_cells / total_cells) × 100
Uniqueness    = (1 - duplicate_rows / total_rows) × 100
Consistency   = % of numeric columns with std > 0
Overall Score = weighted average of the three
```

#### Smart Recommendations
Rule-based action cards — no API call needed. Triggers include:

| Rule | Recommendation |
|---|---|
| Column > 20% missing | Handle Missing Data — suggest imputation |
| Duplicate rows > 0 | Remove Duplicates — run `df.drop_duplicates()` |
| ≥ 2 high correlations | Multicollinearity Alert — suggest PCA/VIF |
| Column > 5% outliers | Outlier Treatment — suggest RobustScaler |
| Binary column detected | Potential Classification Target |
| Zero-variance column | Drop constant column |

> **Screenshot:** Recommendation cards with colour-coded borders

#### Data Quality Radar
A spider/radar chart with 5 axes — Completeness, Uniqueness, Consistency, No Outliers, No Duplicates — giving a visual quality fingerprint of the dataset.

#### Dataset Preview
A scrollable dataframe showing the first N rows (controlled by the sidebar slider).

#### Column Summary
Table showing each column's name, data type, missing count, missing %, and number of unique values.

#### Descriptive Statistics
Standard `df.describe()` output — count, mean, std, min, 25%, 50%, 75%, max for all numeric columns.

#### Strong Correlations
Table of feature pairs with Pearson |r| ≥ 0.7 — flags potential multicollinearity.

#### Detected Outliers
IQR-based outlier table per column showing count, percentage, valid range, and extreme values.

**IQR Method:**
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
Outlier = any value outside [Lower, Upper]
```

#### Feature Importance
Variance-based ranking showing which numeric columns carry the most signal.

#### Downloads
- **Download Dataset (CSV)** — processed version of the uploaded file
- **Download Analysis Report (.txt)** — full text export of the profile + any AI insights generated

---

### 4.4 Visualizations Tab

A dashboard-style layout with 7 chart types, all built with Plotly for full interactivity (zoom, pan, hover tooltips, download PNG).

#### Missing Values Chart
Horizontal bar chart showing the percentage of null cells per column. Only shown if missing values exist.

> **Screenshot:** Horizontal bar chart — columns on Y axis, % missing on X axis

#### Feature Distributions (Histograms)
One histogram per numeric column, each with a unique colour from the palette. Uses 20 bins and semi-transparent fills with white bar borders for readability.

> **Screenshot:** Grid of histograms, one per numeric feature, colour-coded

#### Correlation Heatmap
A full Pearson correlation matrix rendered as a colour-coded heatmap (blue = negative, red = positive). Hovering shows the exact correlation value.

> **Screenshot:** Square heatmap with colour scale

#### Boxplots
One boxplot per numeric column showing median, IQR, whiskers, and individual outlier points. Semi-transparent fills with solid colour borders.

> **Screenshot:** Row of boxplots showing distribution spread and outliers

#### Categorical Distributions
One bar chart per categorical column showing value counts. Labels are auto-truncated to prevent overflow. Count values displayed outside each bar.

> **Screenshot:** Horizontal bar charts for categorical columns

#### Feature Importance Chart
Horizontal bar chart ranking numeric features by variance share (or by Pearson |r| if a target column is detected).

#### Correlation Network
A circular force-layout graph where:
- **Nodes** = columns (sized by number of strong connections)
- **Blue edges** = strong positive correlation (|r| ≥ 0.7)
- **Rose edges** = strong negative correlation

> **Screenshot:** Network graph with nodes arranged in a circle, edges between correlated features

---

### 4.5 AI Insights Tab

Sends a structured prompt to **Groq API (Llama 3.3-70B)** and returns a markdown-formatted analysis with 5 sections:

| Section | Content |
|---|---|
| Key Insights | 3–5 most important patterns and relationships |
| Potential Trends | 2–3 visible or strongly implied trends |
| Anomalies & Quality Issues | Outliers, skewed distributions, data entry errors |
| Suggested Business Questions | 3–4 questions stakeholders should explore |
| Recommended Next Steps | Feature engineering, transformations, next analyses |

The prompt includes: dataset shape, column types, descriptive stats, high correlations, categorical distributions, outlier summary, and missing values. This ensures the AI gives **dataset-specific** answers, not generic advice.

> **Screenshot:** Dark card with markdown-formatted AI response

**Prompt Structure:**
```
You are an expert data scientist.

## Dataset Overview
- Shape: 891 rows × 12 columns
- Numeric columns (5): Age, Fare, ...
- Categorical columns (5): Sex, Embarked, ...
- Total missing values: 177

## Data Quality Score
- Overall: 72/100

## Descriptive Statistics
...

## Strong Correlations
...

## Detected Outliers
...

---
Provide:
### 1. Key Insights
### 2. Potential Trends
### 3. Anomalies & Data Quality Issues
### 4. Suggested Business Questions
### 5. Recommended Next Steps
```

---

### 4.6 Data Story Tab

Generates a **3-paragraph plain-English narrative** from the AI insights — no bullet points, no jargon. Designed for:
- Executive summaries
- Non-technical stakeholders
- Assignment reports

The story follows a fixed structure:
1. **Para 1** — What the dataset represents and its scope
2. **Para 2** — The 2–3 most interesting patterns in plain English
3. **Para 3** — 2–3 clear, actionable recommendations

> **Screenshot:** Warm amber-toned card with italic narrative text

> Note: Data Story requires AI Insights to be generated first.

---

## 5. Technical Deep Dive

### Caching Strategy
```python
@st.cache_data(show_spinner=False)
def _load_df(file_bytes: bytes) -> pd.DataFrame:
    ...

@st.cache_data(show_spinner=False)
def _run_analysis(file_bytes: bytes) -> tuple:
    ...
```
`file_bytes: bytes` is used as the cache key (hashable). The entire analysis pipeline runs once per unique file — re-renders and tab switches are instant.

### Session State
AI-generated content is stored in `st.session_state` keyed by filename:
```python
st.session_state["insights"]       # AI insights text
st.session_state["insights_file"]  # filename it was generated for
st.session_state["data_story"]     # data story text
st.session_state["story_file"]     # filename it was generated for
```
This means switching tabs or adjusting sliders never re-runs the API call.

### Chart Colour System
```python
_PALETTE = ["#4f46e5","#0891b2","#059669","#d97706",
            "#dc2626","#7c3aed","#0d9488","#db2777",
            "#65a30d","#ea580c"]

def _rgba(hex_color: str, alpha: float) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha)"""
```
Each chart uses a unique palette colour. `_rgba()` creates semi-transparent fills for histograms and boxplots.

### Label Clipping Prevention
```python
def _label_margin(labels: list) -> dict:
    """Auto-compute left margin from longest label length."""
    max_len = max((len(str(l)) for l in labels), default=10)
    margin = max(140, min(300, max_len * 7))
    return dict(l=margin)
```
Prevents long categorical labels from being clipped on the left axis.

---

## 6. Data Flow

```
User uploads CSV
        │
        ▼
file_bytes = uploaded_file.read()
        │
        ├──► _load_df(file_bytes)
        │         └──► pd.read_csv() → DataFrame
        │
        └──► _run_analysis(file_bytes)
                  ├──► profile_dataset(df)
                  │         ├── shape, dtypes, missing
                  │         ├── descriptive stats
                  │         ├── correlations
                  │         ├── categorical counts
                  │         └── memory usage
                  │
                  ├──► compute_dataset_quality_score(df)
                  │         ├── completeness %
                  │         ├── uniqueness %
                  │         ├── consistency %
                  │         └── overall score /100
                  │
                  ├──► detect_anomalies(df)
                  │         └── IQR method per numeric column
                  │
                  └──► compute_feature_importance(df)
                            └── variance share ranking

[User clicks Generate AI Insights]
        │
        ▼
_build_insights_prompt(profile, quality, anomalies)
        │
        ▼
Groq API → llama-3.3-70b-versatile
        │
        ▼
Markdown response → st.session_state["insights"]
        │
        ▼
[User clicks Generate Data Story]
        │
        ▼
_build_story_prompt(profile, insights)
        │
        ▼
Groq API → llama-3.3-70b-versatile
        │
        ▼
3-paragraph narrative → st.session_state["data_story"]
```

---

## 7. AI Integration

### Model
**Groq API — `llama-3.3-70b-versatile`**
- Free tier with generous rate limits
- ~500 tokens/sec inference speed
- 128k context window

### Why Groq over OpenAI/Anthropic?
| | Groq | OpenAI GPT-4o | Anthropic Claude |
|---|---|---|---|
| Cost | Free | ~$5/1M tokens | ~$15/1M tokens |
| Speed | ~500 tok/s | ~50 tok/s | ~80 tok/s |
| Context | 128k | 128k | 200k |
| Requires billing | No | Yes | Yes |

### Prompt Engineering Decisions
- **Dataset-specific context**: Every prompt includes actual column names, real statistics, and real outlier values — the model cannot give generic answers
- **Structured output format**: Sections are explicitly numbered and labelled so the response renders cleanly as markdown
- **Token cap**: Insights capped at 2048 tokens, story at 1024 — fast responses, no runaway costs
- **Descriptive stats truncated**: Stats table included in full; categorical counts limited to top 6 columns × top 5 values to stay within context

---

## 8. Tech Stack

| Layer | Technology | Version |
|---|---|---|
| UI Framework | Streamlit | ≥ 1.32 |
| Data Manipulation | Pandas | ≥ 2.0 |
| Numerical Computing | NumPy | ≥ 1.24 |
| Interactive Charts | Plotly | ≥ 5.18 |
| AI Inference | Groq SDK | ≥ 0.9 |
| AI Model | Llama 3.3-70B (via Groq) | — |
| Language | Python | 3.11+ |
| Hosting | Streamlit Community Cloud | — |

---

*Built by Himanshu Mishra · [datalens-ai.streamlit.app](https://datalens-ai.streamlit.app)*
