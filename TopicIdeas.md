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
| 1 | | | | |
| 2 | | | | |

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

## Notes
- Each member should add **at least 2 topic ideas** with a brief summary, data source, and potential methods.
- We'll discuss and pick one topic for the final proposal.
- Remember: the proposal needs a clear **research question**, a **data set**, and a **statistical analysis plan**.
