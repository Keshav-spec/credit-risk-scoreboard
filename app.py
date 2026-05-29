import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

 
# PAGE CONFIG
 

st.set_page_config(
    page_title="Credit Risk Monitor",
    layout="wide"
)

 
# LOAD DATA
 

df = pd.read_csv(
    "data/features/monthly_scores.csv"
)

val = pd.read_csv(
    "outputs/reports/validation_report.csv"
)

 
# TITLE
 

st.title("📊 Credit Risk Scorecard Monitoring Dashboard")

st.markdown("""
Production-style monitoring dashboard for:
- Credit score drift
- Model stability
- Portfolio risk segmentation
- PSI monitoring
""")

st.markdown("---")

 

# SIDEBAR
 

st.sidebar.header("Filters")

selected_month = st.sidebar.selectbox(
    "Select Month",
    sorted(df["month"].unique())
)

filtered_df = df[
    df["month"] == selected_month
]


# EXECUTIVE SUMMARY
 

st.subheader("Executive Portfolio Summary")

total_loans = len(filtered_df)

avg_score = filtered_df["credit_score"].mean()

default_rate = (
    filtered_df["target"].mean() * 100
)

high_risk_pct = (
    (
        filtered_df["score_band"]
        .isin(["Very High Risk", "High Risk"])
    ).mean() * 100
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Loans",
    f"{total_loans:,}"
)

c2.metric(
    "Average Score",
    f"{avg_score:.0f}"
)

c3.metric(
    "Default Rate",
    f"{default_rate:.2f}%"
)

c4.metric(
    "High Risk Exposure",
    f"{high_risk_pct:.1f}%"
)

st.markdown("---")

 
# APPROVAL STRATEGY SIMULATOR
 

st.subheader("Approval Strategy Simulator")

cutoff = st.slider(
    "Minimum Credit Score for Approval",
    300,
    850,
    650
)

approved = filtered_df[
    filtered_df["credit_score"] >= cutoff
]

approval_rate = (
    len(approved)
    / len(filtered_df)
) * 100

expected_default = (
    approved["target"].mean() * 100
    if len(approved) > 0
    else 0
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Approval Rate",
    f"{approval_rate:.1f}%"
)

c2.metric(
    "Expected Default Rate",
    f"{expected_default:.2f}%"
)

c3.metric(
    "Approved Loans",
    f"{len(approved):,}"
)

fig_cutoff = px.histogram(
    filtered_df,
    x="credit_score",
    color="target",
    nbins=30
)

fig_cutoff.add_vline(
    x=cutoff,
    line_dash="dash",
    annotation_text=f"Cutoff = {cutoff}"
)

st.plotly_chart(
    fig_cutoff,
    use_container_width=True
)

st.markdown("---")



 
# KPI SECTION
 

st.subheader("Model Validation Metrics")

c1, c2, c3 = st.columns(3)

gini = val[val["Metric"] == "Gini"]["Value"].values[0]
ks = val[val["Metric"] == "KS"]["Value"].values[0]
psi = val[val["Metric"] == "PSI"]["Value"].values[0]

c1.metric(
    "Gini",
    f"{gini:.3f}",
    delta="Pass ✓" if gini > 0.4 else "Fail ✗"
)

c2.metric(
    "KS Statistic",
    f"{ks:.3f}",
    delta="Pass ✓" if ks > 0.3 else "Fail ✗"
)

c3.metric(
    "PSI",
    f"{psi:.3f}",
    delta="Stable ✓" if psi < 0.1 else "Drift ⚠"
)

st.markdown("---")


 
# RISK ALERTS
 

st.subheader("Risk Monitoring Alerts")

alerts = []

if psi > 0.1:
    alerts.append("⚠ PSI threshold exceeded")

if ks < 0.3:
    alerts.append("⚠ KS below acceptable threshold")

if gini < 0.4:
    alerts.append("⚠ Gini deterioration detected")

if default_rate > 15:
    alerts.append("⚠ Portfolio default rate elevated")

if alerts:

    for a in alerts:
        st.warning(a)

else:
    st.success("✅ All monitoring metrics within thresholds")

st.markdown("---")


 
# SCORE DISTRIBUTION
 

st.subheader("Credit Score Distribution")

fig_hist = px.histogram(
    filtered_df,
    x="credit_score",
    color="target",
    nbins=30,
    opacity=0.7,
    barmode="overlay"
)

fig_hist.update_layout(
    xaxis_title="Credit Score",
    yaxis_title="Loan Count"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

 
# SCORE DRIFT OVER TIME
 

st.subheader("Monthly Score Drift")

fig_box = go.Figure()

months = sorted(df["month"].unique())

for m in months:

    sub = df[
        df["month"] == m
    ]["credit_score"]

    fig_box.add_trace(
        go.Box(
            y=sub,
            name=m,
            boxmean=True
        )
    )

fig_box.update_layout(
    yaxis_title="Credit Score"
)

st.plotly_chart(
    fig_box,
    use_container_width=True
)


 
# MONTHLY RISK TRENDS
 

st.subheader("Portfolio Risk Trends")

trend_df = (
    df.groupby("month")
    .agg(
        avg_score=("credit_score", "mean"),
        default_rate=("target", "mean")
    )
    .reset_index()
)

trend_df["default_rate"] *= 100

fig_trend = go.Figure()

fig_trend.add_trace(
    go.Scatter(
        x=trend_df["month"],
        y=trend_df["avg_score"],
        mode="lines+markers",
        name="Average Score"
    )
)

fig_trend.add_trace(
    go.Scatter(
        x=trend_df["month"],
        y=trend_df["default_rate"],
        mode="lines+markers",
        name="Default Rate (%)",
        yaxis="y2"
    )
)

fig_trend.update_layout(
    yaxis=dict(title="Average Score"),
    yaxis2=dict(
        title="Default Rate %",
        overlaying="y",
        side="right"
    )
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)

st.markdown("---")
 
# SCORE BAND ANALYSIS
 

st.subheader("Portfolio Risk Segmentation")

band_summary = (
    filtered_df
    .groupby("score_band")
    .agg(
        loans=("credit_score", "count"),
        avg_score=("credit_score", "mean")
    )
    .reset_index()
)

fig_bar = px.bar(
    band_summary,
    x="score_band",
    y="loans",
    color="avg_score",
    text="loans"
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)

 
# PSI ALERT TABLE
 

st.subheader("Monthly PSI Monitoring")

ref = df[
    df["month"] == "2025-01"
]["credit_score"]

psi_rows = []

for m in months[1:]:

    cur = df[
        df["month"] == m
    ]["credit_score"]

    ref_hist = (
        np.histogram(ref, 10)[0]
        / len(ref)
    )

    cur_hist = (
        np.histogram(cur, 10)[0]
        / len(cur)
    )

    ref_hist = ref_hist + 1e-4
    cur_hist = cur_hist + 1e-4

    psi_value = np.sum(
        (cur_hist - ref_hist)
        * np.log(cur_hist / ref_hist)
    )

    status = (
        "🟢 Stable"
        if psi_value < 0.1
        else "🟡 Monitor"
        if psi_value < 0.2
        else "🔴 Retrain"
    )

    psi_rows.append({
        "Month": m,
        "PSI": round(psi_value, 4),
        "Status": status
    })

psi_df = pd.DataFrame(psi_rows)

st.dataframe(
    psi_df,
    use_container_width=True,
    hide_index=True
)

 
# INDIVIDUAL LOAN EXPLORER
 

st.subheader("Individual Loan Risk Explorer")

sample_idx = st.slider(
    "Select Loan Record",
    0,
    len(filtered_df) - 1,
    0
)

loan = filtered_df.iloc[sample_idx]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Credit Score",
    int(loan["credit_score"])
)

c2.metric(
    "Risk Band",
    loan["score_band"]
)

c3.metric(
    "Probability of Default",
    f"{loan['prob_default'] * 100:.2f}%"
)

recommendation = (
    "Approve"
    if loan["credit_score"] >= cutoff
    else "Reject"
)

c4.metric(
    "Recommendation",
    recommendation
)

st.caption(
    "Built with Python, Streamlit, Plotly, SQLite, WoE Encoding, Logistic Regression, and PSI-based monitoring."
)