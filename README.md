# Credit Risk Scorecard & Model Monitoring System

An end-to-end bank-grade credit risk modelling and monitoring system built on the LendingClub dataset using Weight of Evidence (WoE) encoding, logistic regression, and industry-standard validation metrics including Gini coefficient, KS statistic, and Population Stability Index (PSI).

The project delivers a calibrated credit scorecard on a 300–850 scale, SQL-based portfolio analysis, an interactive Tableau dashboard, and a Streamlit monitoring platform with drift detection and approval strategy simulation.

**Live Dashboard (Tableau):** [Credit-Risk-Score](https://public.tableau.com/views/CreditRiskScore/Dashboard1?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)  
**Live App (Streamlit):** [credit-risk-scoreboard.streamlit.app](https://credit-risk-scoreboard.streamlit.app/) 
**Dataset:** LendingClub Accepted Loans 2007–2018 Q4 via Kaggle (CC0 Public Domain)  
**Author:** Keshav Sharma — CSE with Data Science

---

## Dashboard Preview

**Screenshot 1 — Tableau: Score distribution and default rate by band**

![Tableau Dashboard](screenshots/tableau_dashboard.png)

**Screenshot 2 — Streamlit: Model monitoring with PSI alerts**

![Streamlit Monitoring Dashboard](screenshots/streamlit_monitoring.png)

**Screenshot 3 — Validation: ROC curve and KS chart**

![Validation Charts](screenshots/roc_ks_chart.png)

---

## Project Overview

Credit risk modelling is one of the most important applications of machine learning in financial services. Unlike most academic projects that stop at model training and accuracy reporting, this project implements the full lifecycle of a production-style credit scorecard system used by banks and NBFCs.

The emphasis is on interpretability, governance, deployability, and business usability — not leaderboard optimisation.

The pipeline covers:

- Leakage-free feature selection from 150+ raw columns
- WoE/IV-based feature engineering
- Interpretable scorecard modelling with logistic regression
- Industry-standard validation and governance metrics
- SQL-based portfolio analytics
- Tableau dashboard for portfolio reporting
- Streamlit monitoring platform with drift detection and approval simulation

---

## End-to-End Pipeline

```
Raw LendingClub Data
        |
        v
Cleaning & Leakage Removal
        |
        v
Feature Selection (IV filtering)
        |
        v
WoE Binning & IV Analysis
        |
        v
Logistic Regression Training
        |
        v
Scorecard Calibration (PDO)
        |
        v
Validation (Gini / KS / PSI)
        |
        v
SQL Portfolio Analytics
        |
        v
Tableau Dashboard
        |
        v
Streamlit Monitoring Platform
```

---

## Key Results

| Metric | Value | Target | Status |
|---|---|---|---|
| Train AUC | 0.701 | Above 0.70 | Pass |
| Test AUC | 0.701 | Above 0.70 | Pass |
| Gini Coefficient | 0.402 | Above 0.40 | Pass |
| KS Statistic | 0.293 | Above 0.30 | Near target |
| PSI | 0.002 | Below 0.10 | Stable |

### Score Distribution Summary

| Score Band | Score Range | Portfolio Segment |
|---|---|---|
| Very High Risk | 300 to 399 | High probability of default |
| High Risk | 400 to 499 | Elevated default risk |
| Medium Risk | 500 to 599 | Moderate credit quality |
| Low Risk | 600 to 699 | Strong borrower profile |
| Very Low Risk | 700 to 850 | Lowest expected default risk |

---

## What Makes This Project Different

Most student credit risk projects train a model and report accuracy. This system goes further by implementing:

- WoE and IV-based scorecard methodology used in production banking systems
- Leakage prevention and governance-aware feature engineering
- Score calibration to a FICO-style 300–850 points system
- PSI-based monitoring for population drift detection
- SQL portfolio analytics layer for business-level insight
- Tableau dashboards for portfolio reporting
- Streamlit-based monitoring platform with approval strategy simulation
- Interpretable modelling focused on regulatory explainability and auditability

---

## Business Impact

This system simulates a production-grade retail credit risk workflow used by banks and lending institutions for:

- Borrower risk segmentation
- Approval threshold optimisation
- Portfolio monitoring
- Early-warning drift detection
- Model governance and retraining decisions

The approval strategy simulator allows lenders to evaluate the trade-off between approval rate, expected default rate, and portfolio risk exposure at different score cut-offs.

---

## Methodology

### Why Logistic Regression Instead of Tree-Based Models

WoE encoding transforms each feature bin into a value that has a linear relationship with log-odds by construction. Logistic regression is the natural fit for this input representation because coefficients remain interpretable, feature contributions are explainable, and the final score can be decomposed into individual point contributions per feature bin.

Banks and regulators (RBI, Basel II/III) require that lending decisions be explainable to customers and auditors. A logistic regression scorecard satisfies this requirement. A gradient-boosted tree does not.

### Weight of Evidence (WoE)

WoE replaces each categorical or binned continuous value with the natural log of the ratio of good to bad accounts in that bin:

```
WoE = ln( % Good in bin / % Bad in bin )
```

| WoE Value | Interpretation |
|---|---|
| Positive | Lower-than-average default risk |
| Near zero | Average population risk |
| Negative | Higher-than-average default risk |

Benefits over one-hot encoding: handles missing values as a natural bin, stabilises input distributions, reduces dimensionality, and improves logistic regression fit.

### Information Value (IV)

IV summarises the total predictive power of a feature across all its bins:

```
IV = sum over bins of ( (%Good - %Bad) x WoE )
```

| IV Range | Interpretation | Decision |
|---|---|---|
| Below 0.02 | Useless predictor | Dropped |
| 0.02 to 0.10 | Weak predictor | Dropped |
| 0.10 to 0.30 | Medium predictor | Retained |
| 0.30 to 0.50 | Strong predictor | Retained |
| Above 0.50 | Potential leakage | Reviewed |

Only medium and strong predictors were retained.

### Scorecard Calibration

Model probabilities were converted into a FICO-style integer score using the Points to Double Odds (PDO) methodology.

Parameters used:

| Parameter | Value |
|---|---|
| Base score | 600 |
| PDO | 50 |
| Score range | 300 to 850 |

This converts raw default probabilities into business-friendly credit scores that non-technical stakeholders can interpret and act on.

### Validation Metrics

**Gini Coefficient** measures the model's ability to rank borrowers from least to most risky. Gini = 2 x AUC - 1. Target: above 0.40.

**KS Statistic** measures the maximum separation between the cumulative distributions of good and bad accounts across all score thresholds. It identifies the optimal decision cut-off. Target: above 0.30.

**Population Stability Index (PSI)** measures whether the score distribution has shifted between the development sample and a later monitoring window.

| PSI Value | Status | Action |
|---|---|---|
| Below 0.10 | Stable | No action required |
| 0.10 to 0.25 | Minor shift | Investigate |
| Above 0.25 | Major shift | Retrain model |

---

## Data Leakage Prevention

The LendingClub dataset contains columns that are only available after a loan has been issued — total payments received, recovery amounts, last payment dates, and others. Including these in training produces artificially high AUC and a model that cannot be deployed in practice.

Two leakage controls were applied:

**Column-level:** All post-origination variables were excluded. Only application-time features were used, including FICO score, debt-to-income ratio, employment length, and delinquency history.

**Feature engineering level:** WoE bins were fitted exclusively on the training set. Test data was transformed using frozen bin boundaries. This prevents leakage from the encoding stage, which is a commonly overlooked source of inflated performance in student projects.

---

## Features Used

| Feature | Type | Purpose |
|---|---|---|
| `fico_range_low` | Numeric | Core creditworthiness signal |
| `dti` | Numeric | Debt burden relative to income |
| `annual_inc` | Numeric | Repayment capacity |
| `loan_amnt` | Numeric | Exposure size |
| `int_rate` | Numeric | Lender's assessed risk premium |
| `grade` / `sub_grade` | Categorical | LendingClub's internal risk assessment |
| `emp_length` | Categorical | Employment stability signal |
| `home_ownership` | Categorical | Asset and financial stability indicator |
| `purpose` | Categorical | Loan intent risk segmentation |
| `inq_last_6mths` | Numeric | Recent credit-seeking behaviour |
| `delinq_2yrs` | Numeric | Historical delinquency signal |
| `revol_util` | Numeric | Credit utilisation ratio |
| `open_acc` | Numeric | Breadth of credit relationships |
| `verification_status` | Categorical | Income verification level |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data source | LendingClub via Kaggle | 2.2M accepted loan records, 2007–2018 |
| Data processing | pandas, numpy | Cleaning, feature selection, sampling |
| WoE encoding | scorecardpy, optbinning | Binning, IV calculation, WoE transformation |
| Modelling | scikit-learn | Logistic regression, train/test split |
| Scorecard calibration | scorecardpy | PDO-based points conversion |
| Validation | numpy, matplotlib | Gini, KS, PSI computation and charting |
| Database | SQLite | Scored loan storage and querying |
| SQL analytics | SQLite + pandas | Portfolio business analysis |
| Dashboard | Tableau Public | Portfolio scorecard visualisation |
| Monitoring platform | Streamlit, Plotly | PSI alerts, drift tracking, approval simulation |
| Version control | Git, GitHub | Source code and pipeline management |

---

## Project Structure

```
credit-risk-scorecard/
|
|-- data/
|   |-- raw/                         # LendingClub CSV — downloaded separately (see Setup)
|   |-- processed/
|   |   `-- loans_clean.csv
|   `-- features/
|       |-- train_woe.csv
|       |-- test_woe.csv
|       |-- test_scored.csv
|       `-- monthly_scores.csv
|
|-- models/
|   |-- woe_bins.pkl
|   `-- logistic_model.pkl
|
|-- outputs/
|   |-- plots/
|   |   |-- default_by_grade.png
|   |   |-- fico_boxplot.png
|   |   |-- correlation_heatmap.png
|   |   `-- roc_ks_chart.png
|   `-- reports/
|       |-- iv_table.csv
|       |-- scorecard_table.csv
|       `-- validation_report.csv
|
|-- sql/
|   `-- analysis_queries.sql
|
|-- screenshots/
|
|-- 01_data_cleaning.py
|-- 02_eda.py
|-- 03_woe_encoding.py
|-- 04_model_training.py
|-- 05_scorecard.py
|-- 06_validation.py
|-- 10_model_monitoring.py
|-- app.py
|-- requirements.txt
`-- README.md
```

---

## Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-scorecard.git
cd credit-risk-scorecard
```

**2. Create and activate virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Go to [kaggle.com](https://www.kaggle.com) and search for "All Lending Club Loan Data" by wordsforthewise. Download `accepted_2007_to_2018Q4.csv.gz` and place it inside `data/raw/`. No manual extraction is required — pandas reads the gzip file directly.

**5. Run the pipeline in order**

```bash
python 01_data_cleaning.py
python 02_eda.py
python 03_woe_encoding.py
python 04_model_training.py
python 05_scorecard.py
python 06_validation.py
python 10_model_monitoring.py
```

Each script reads outputs from the previous step. Do not skip steps.

**6. Launch the monitoring dashboard**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## SQL Analysis Layer

The file `sql/analysis_queries.sql` contains four analytical queries on the scored dataset:

- Default rate and average loan amount by score band
- Average credit score versus actual default rate by loan grade
- Portfolio-level score percentiles and approval rate at a 650 score threshold
- High-risk large loans — loans above 20,000 with scores below 450, ranked by exposure

These queries translate model outputs into portfolio management decisions, demonstrating business-layer thinking beyond the modelling pipeline.

---

## Model Monitoring

The Streamlit application simulates six months of post-deployment score data with gradual distribution drift introduced in later months. For each month it computes PSI against the development baseline and flags the population stability status.

Features of the monitoring platform:

- Monthly score drift tracking with boxplot visualisation
- PSI alerts with traffic-light status (Stable / Monitor / Retrain)
- Approval strategy simulator — evaluates approval rate vs expected default rate at different score cut-offs
- Portfolio risk segmentation by score band
- Validation KPI dashboard (Gini, KS, PSI)
- Borrower-level score explorer

This simulates the model governance workflows used by risk teams at banks and lending institutions, where models are monitored monthly and flagged for redevelopment when stability thresholds are breached.

---

## Limitations

- The dataset contains only approved loans, introducing potential sample-selection bias. A reject inference module would be needed to correct for this in production.
- Monthly score drift is simulated rather than sourced from live production data.
- Logistic regression prioritises interpretability and regulatory compliance over maximum predictive power.

---

## Future Improvements

- Characteristic Stability Index (CSI) monitoring per feature, not just overall PSI
- Reject inference modelling to address sample-selection bias
- Live API-based monitoring once a suitable data source is available
- Multi-model comparison: WoE logistic regression versus XGBoost with SHAP explainability
- Automated retraining pipeline triggered by PSI breach
- Loss-given-default (LGD) and exposure-at-default (EAD) modelling for full expected loss estimation

---

## Author

**Keshav Sharma** — CSE with Data Science  
GitHub: [github.com/Keshav-spec](https://github.com/Keshav-spec)  
LinkedIn: [linkedin.com/in/keshav-sharma](https://www.linkedin.com/in/keshav-sharma-0396b1287/)

---

## License

This project is licensed under the [MIT License].

The LendingClub dataset is released under the CC0 Public Domain licence. This project is intended for educational and portfolio purposes only and does not constitute financial advice.