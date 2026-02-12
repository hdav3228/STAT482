# Topic Ideas — Group 10

**Team Members:** Harsh Dave, Nick Griffin, Jack Leidlein, Hector Carrillo

---

## Harsh Dave

| # | Topic | Summary | Data Source | Methods / Notes |
|---|-------|---------|-------------|-----------------|
| 1 | **AI-Powered Data Breach Hub** | Aggregating, classifying, and normalizing public data breach reports using AI. Tracks breach trends across sectors (universities, healthcare, etc.) and provides real-time intelligence via a Streamlit dashboard. Originally built with GenAI scrapers, AWS services, and Elasticsearch. | Publicly available breach disclosures and security incident reports (no PII). | Classification models, NLP/GenAI for scraping and categorization, trend analysis across sector/country/severity. Could pivot to a statistical analysis of breach trends, severity prediction, or sector vulnerability modeling. |
| 2 | **Modeling Swing and Miss on MLB Pitches** | Logistic regression analysis of what makes a pitch hard to hit, using Statcast pitch-by-pitch data from the 2025 season (2-strike counts, ~97K obs). Features: release speed, horizontal break, induced vertical break, release extension, arm angle + all 2-way interactions. Found extension heavily increases whiff chance; speed × movement interactions are the primary drivers of whiff%. | Statcast via `pybaseball` Python module — pitch-level data for every MLB pitch. | Logistic regression with backward selection (α = 0.05), interaction plots, deviance residual diagnostics. Could extend with spatial modeling, random effects for pitcher/batter, or pitch-sequence effects. |
| 3 | | | | |
| 4 | | | | |

---

## Nick Griffin

| # | Topic | Summary | Data Source | Methods / Notes |
|---|-------|---------|-------------|-----------------|
| 1 | **Predicting Stock Volatility Across Sectors** | Model daily stock price volatility (e.g., intraday range or rolling standard deviation of returns) as a function of sector, trading volume, and lagged price features using 120 major US companies. Investigate whether certain sectors exhibit systematically higher volatility and how volume shocks propagate across time. | [US Stock Market Historical OHLCV Dataset](https://www.kaggle.com/datasets/asadullahcreative/us-stock-market-historical-ohlcv-dataset) — 184K daily records, 120 companies, 9 sectors. | Time-series regression, GARCH-type volatility modeling, sector fixed effects, rolling-window feature engineering. Could extend with cross-sector correlation analysis or regime-switching models. |
| 2 | **Predicting Heart Disease Risk from Patient Health Indicators** | Build a classification model to predict the presence of heart disease using demographic, behavioral, and clinical features (BMI, smoking status, physical activity, diabetes, etc.) from a large CDC survey sample. Explore which risk factors are the strongest predictors and whether interactions (e.g., age × diabetes) improve prediction. | [Heart Disease Health Indicators — BRFSS 2015](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset) — ~250K survey respondents, 22 features, binary outcome. | Logistic regression with variable selection, ROC/AUC evaluation, interaction terms, possible comparison with tree-based methods (random forest, gradient boosting). |

---

## Jack Leidlein

| # | Topic | Summary | Data Source | Methods / Notes |
|---|-------|---------|-------------|-----------------|
| 1 | | | | |
| 2 | | | | |

---

## Hector Carrillo

| # | Topic | Summary | Data Source | Methods / Notes |
|---|-------|---------|-------------|-----------------|
| 1 | | | | |
| 2 | | | | |

---
