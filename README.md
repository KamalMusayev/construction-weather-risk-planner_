# Construction Weather Risk Planner

**Weather-risk intelligence for construction operations**

Construction Weather Risk Planner is an end-to-end data and machine learning project that converts weather data into actionable construction-risk decisions. Instead of showing only raw weather forecasts, the system estimates whether construction activities may be affected by wind, freezing conditions, heat stress, or heavy rainfall.

The project includes a data ingestion pipeline, DuckDB database layers, feature engineering, predictive modeling, rule-based risk scoring, SHAP explainability, 16-day forecast generation, and a Streamlit dashboard.

---

## Project Goal

Construction teams often have access to weather forecasts, but those forecasts do not directly answer operational questions such as:

- Can crane operations continue safely?
- Is concrete work exposed to freezing risk?
- Are workers exposed to heat-stress conditions?
- Should excavation or low-area work be delayed because of rainfall?

This project translates weather conditions into construction-specific risk outputs that are easier to use for planning and safety decisions.

---

## Key Features

- **Automated weather ingestion** from Open-Meteo
- **Historical + forecast pipeline** using DuckDB
- **Four-city coverage:** Baku, Ganja, Nakhchivan, Shusha
- **Crane risk prediction** using machine learning
- **Freeze, heat, and flood risk scoring** using domain-driven rules
- **16-day forecast output** for each city
- **SAFE / WARNING / DANGER alert categories**
- **SHAP explainability** for the crane-risk model
- **Interactive Streamlit dashboard** for city/day risk exploration

---

## System Overview

```text
Open-Meteo API
     ↓
Raw CSV Files
     ↓
DuckDB Raw Layer
     ↓
DuckDB Staging Layer
     ↓
Analytics / Feature Engineering
     ↓
Crane ML Model + Rule-Based Risk Engine
     ↓
16-Day Risk Forecast CSV
     ↓
Streamlit Dashboard
```

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                         # Historical and forecast CSV files
│   ├── database/weather_daily.duckdb # DuckDB database
│   └── predictions/risk_forecast.csv # Final dashboard-ready forecast output
├── models/
│   ├── crane_logreg.pkl              # Logistic Regression baseline model
│   ├── crane_xgb_calibrated.pkl      # XGBoost crane-risk model
│   ├── feature_list.pkl              # Model feature list
│   ├── scaler.pkl                    # Scaler for logistic regression
│   └── thresholds.pkl                # Decision thresholds
├── notebooks/
│   ├── day_08_modeling.ipynb         # Predictive modeling and evaluation
│   └── day_09_shap_explainability.ipynb # SHAP model explanation
├── reports/figures/                  # Charts, evaluation plots, SHAP plots
├── src/
│   ├── app.py                        # Streamlit dashboard
│   ├── ingestion.py                  # Open-Meteo ingestion logic
│   ├── cleaning.py                   # Raw-to-staging cleaning
│   ├── database.py                   # DuckDB schema and loading logic
│   ├── features.py                   # Feature engineering layer
│   ├── pipeline.py                   # Full / incremental pipeline runner
│   └── quality_checks.py             # Data validation checks
├── requirements.txt
└── README.md
```

---

## Cities Covered

| City | Latitude | Longitude | Main Weather-Relevant Context |
|---|---:|---:|---|
| Baku | 40.41 | 49.87 | Coastal wind and humidity exposure |
| Ganja | 40.68 | 46.36 | Inland seasonal variability |
| Nakhchivan | 39.21 | 45.41 | Dry continental climate and heat extremes |
| Shusha | 39.75 | 46.75 | Mountain climate, cold exposure, precipitation variation |

---

## Risk Types

| Risk | Method | Main Input Signal | Construction Meaning |
|---|---|---|---|
| Crane Risk | Machine Learning | Wind speed, wind gusts, wind interactions | Unsafe lifting or crane operation conditions |
| Freeze Risk | Rule-based probability | Minimum temperature | Concrete curing and material freezing risk |
| Heat Risk | Rule-based probability | Apparent maximum temperature | Worker heat-stress risk |
| Flood Risk | Rule-based probability | Daily precipitation | Excavation, drainage, and low-area flooding risk |

### Risk Alert Categories

| Category | Probability Range | Meaning |
|---|---:|---|
| SAFE | < 40% | Normal operations can continue with standard monitoring |
| WARNING | 40–75% | Increased caution and operational planning are required |
| DANGER | ≥ 75% | High-risk work should be delayed, restricted, or stopped |

---

## Data Sources

The project uses Open-Meteo data for historical weather and forecast weather.

### Historical Data

Used for training, feature engineering, and evaluation.

Core columns include:

- `temperature_2m_max`
- `temperature_2m_min`
- `apparent_temperature_max`
- `apparent_temperature_min`
- `precipitation_sum`
- `precipitation_hours`
- `rain_sum`
- `snowfall_sum`
- `windspeed_10m_max`
- `windgusts_10m_max`
- `weathercode`
- `winddirection_10m_dominant`

### Forecast Data

Used only for inference and dashboard output.

The final dashboard consumes:

```text
data/predictions/risk_forecast.csv
```

---

## Feature Engineering

The modeling notebook creates weather-derived features that improve crane-risk prediction:

| Feature | Purpose |
|---|---|
| `temp_range` | Daily temperature variability |
| `wind_power` | Combined wind speed and gust strength |
| `rain_intensity` | Rainfall normalized by precipitation duration |
| `wind_ratio` | Gust-to-wind ratio, useful for unstable wind conditions |
| `wind_lag1`, `wind_lag2` | Previous wind behavior |
| `wind_roll3` | Short-term wind trend |
| `gust_lag1` | Previous gust exposure |
| `wind_delta` | Day-to-day wind change |
| City one-hot features | Location-specific climate behavior |

---

## Modeling Approach

The project uses a hybrid modeling strategy.

### Crane Risk — Machine Learning

Crane risk is treated as the ML target because wind-related operational risk is not always captured by a single threshold. It depends on wind speed, gust behavior, recent wind trend, and city-specific weather patterns.

Models trained and compared:

- Logistic Regression
- XGBoost Classifier

The stored artifacts are available in:

```text
models/
```

### Freeze / Heat / Flood — Rule-Based Risk Engine

These risks are converted into smooth probabilities from weather signals and then mapped into SAFE / WARNING / DANGER categories. This keeps the system interpretable and aligned with construction decision-making.

---

## Evaluation

Model evaluation outputs are saved in:

```text
reports/figures/model_comparison.csv
reports/figures/model_evaluation.png
```

The current prototype prioritizes interpretability and safety-oriented analysis. For crane-risk prediction, recall and false-negative behavior are especially important because missing a dangerous wind day is more costly than creating a cautious warning.

---

## SHAP Explainability

SHAP analysis was added for the XGBoost crane-risk model to explain which features contribute most to model predictions.

Generated artifacts:

```text
notebooks/day_09_shap_explainability.ipynb
reports/figures/shap_feature_importance.csv
reports/figures/shap_summary_bar.png
reports/figures/shap_summary_beeswarm.png
reports/figures/shap_dependence_*.png
reports/figures/shap_model_metrics.json
```

The SHAP analysis helps answer:

- Which features drive crane-risk predictions?
- Are wind-related features actually important?
- Does the model rely on sensible construction/weather signals?
- Are city effects influencing predictions too much?

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

From the project root:

```bash
python src/pipeline.py --mode full
```

or, if the historical database already exists:

```bash
python src/pipeline.py --mode incremental
```

### 3. Run the dashboard

```bash
streamlit run src/app.py
```

The dashboard displays:

- selected city weather panel
- 16-day forecast date selector
- total construction risk index
- crane / freeze / heat / flood risk cards
- risk-specific action recommendations

---

## Dashboard Output Schema

The final forecast file contains one row per city and forecast day.

Main columns:

| Column | Meaning |
|---|---|
| `date` | Forecast date |
| `city` | City name |
| `day_number` | Forecast day index |
| `crane_prob` | Crane-risk probability |
| `freeze_prob` | Freeze-risk probability |
| `heat_prob` | Heat-risk probability |
| `flood_prob` | Flood-risk probability |
| `total_risk` | Weighted combined risk score |
| `*_alert` | SAFE / WARNING / DANGER label |
| `*_pct` | Probability as percentage |

---

## Limitations

- Forecast quality directly affects risk quality.
- The system currently works at city level, not exact site-coordinate level.
- Freeze, heat, and flood risks use simplified rule-based probability logic.
- Crane model performance should be improved with more labeled data and better validation.
- The dashboard is a prototype and does not yet include user authentication or real-time alerts.

---

## Future Improvements

- Add site-level coordinates instead of city-level only
- Add real-time alert notifications
- Improve crane-risk labels with actual construction incident / operation records
- Add SHAP force plots for single-day decision explanations
- Add model monitoring and drift detection
- Add automated scheduled daily refresh
- Deploy dashboard as a cloud application

---

## Project Summary

Construction Weather Risk Planner turns weather data into operational construction intelligence. The system helps construction teams move from manual weather checking toward proactive, risk-based planning.

> The goal is not only to forecast weather — the goal is to support safer construction decisions.
