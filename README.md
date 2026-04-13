# DataLens AI — AI-Powered Data Analysis Assistant

> Upload any CSV dataset and get instant exploratory data analysis, interactive visualisations, smart recommendations, and AI-generated insights — powered by **Groq / Llama 3** and **Streamlit**. Completely free to run.

---

## Live Demo

[**Open DataLens AI →**](https://datalens-ai.streamlit.app)

---

## Features

| Feature | Details |
|---|---|
| 📋 **Auto Data Profiling** | Shape, column types, missing values, descriptive stats, correlations, duplicate detection |
| 📈 **Interactive Visualisations** | Histograms, correlation heatmap, boxplots, categorical bar charts, missing-value chart (Plotly) |
| 🕸 **Correlation Network** | Force-layout graph showing strongly correlated feature pairs |
| 🏅 **Quality Score** | 0–100 score based on completeness, uniqueness, and consistency |
| 🔍 **Anomaly Detection** | IQR-based outlier detection per numeric column |
| 📊 **Feature Importance** | Variance-based importance ranking (or Pearson \|r\| with a target column) |
| 🎯 **Smart Recommendations** | Rule-based action cards — no API call needed |
| 🤖 **AI Insights** | Llama 3.3-70B analyses your dataset: key findings, trends, anomalies, business questions, next steps |
| 📖 **Data Story** | Plain-English narrative for non-technical stakeholders |
| 📥 **Report Download** | Full analysis exportable as a `.txt` file |

---

## Architecture

```
ai-data-analyst/
├── app.py             ← Streamlit UI · routing · session state · downloads
├── data_analysis.py   ← profile_dataset · detect_anomalies · compute_feature_importance
├── visualization.py   ← Plotly chart builders (histograms, heatmap, boxplots, …)
├── ai_insights.py     ← Groq prompt builders · generate_insights · generate_data_story
├── utils.py           ← Quality score · formatters · report builder · helpers
├── requirements.txt
└── README.md
```

**Data flow**

```
CSV Upload
   │
   ▼
data_analysis.py ──► profile dict ──► visualization.py ──► Plotly charts
                  │                └─► ai_insights.py  ──► Groq API (Llama 3)
                  │                                     └─► markdown insights
                  └─► utils.py ──────────────────────────► text report
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Himaaanshu7/ai-data-analyst.git
cd ai-data-analyst
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Get a **free** key at [console.groq.com](https://console.groq.com) — no credit card required.

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

> The key is loaded automatically — users never need to enter it manually.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example Datasets

| Dataset | Why it's useful |
|---|---|
| [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) | Mixed numeric + categorical, missing values, clear survival patterns |
| [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) | Clean numeric-only, strong inter-feature correlations |
| [Tips](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv) | Small real-world dataset with tip/bill relationships |

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Charts | Plotly |
| Data | Pandas · NumPy |
| AI | Groq API · Llama 3.3-70B (free) |

---

## Project Status

Built as a portfolio-quality data science project by **Himanshu Mishra**.

---

## License

MIT
