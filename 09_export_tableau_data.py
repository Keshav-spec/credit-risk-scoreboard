import pandas as pd
import sqlite3
import os

 # CREATE OUTPUT FOLDER
 
os.makedirs("outputs", exist_ok=True)

 # CONNECT DATABASE
 
print("=" * 60)
print("EXPORTING TABLEAU DATA")
print("=" * 60)

conn = sqlite3.connect(
    "data/credit_risk.db"
)

 # EXPORT MAIN DATASET
 
print("\nExporting main scored dataset...")

main_df = pd.read_sql(
    "SELECT * FROM loan_scores",
    conn
)

main_df.to_csv(
    "outputs/tableau_main.csv",
    index=False
)

print("tableau_main.csv exported.")

 # EXPORT SCORE BAND SUMMARY
 
print("\nExporting score summary dataset...")

summary_df = pd.read_sql("""
SELECT
    score_band,
    COUNT(*) AS total_loans,
    ROUND(AVG(target) * 100, 2) AS default_rate,
    ROUND(AVG(credit_score), 1) AS avg_score,
    ROUND(AVG(loan_amnt), 0) AS avg_loan_amnt
FROM loan_scores
GROUP BY score_band
""", conn)

summary_df.to_csv(
    "outputs/tableau_summary.csv",
    index=False
)

print("tableau_summary.csv exported.")

 # EXPORT VALIDATION METRICS
 
print("\nExporting validation metrics...")

validation_df = pd.read_csv(
    "outputs/reports/validation_report.csv"
)

validation_df.to_csv(
    "outputs/tableau_validation.csv",
    index=False
)

print("tableau_validation.csv exported.")

 # OPTIONAL: PSI MONITORING DATA
 
print("\nCreating PSI monitoring dataset...")

psi_df = pd.read_sql("""
SELECT
    score_band,
    COUNT(*) AS loan_count
FROM loan_scores
GROUP BY score_band
""", conn)

psi_df["dataset"] = "Current"

psi_df.to_csv(
    "outputs/tableau_psi.csv",
    index=False
)

print("tableau_psi.csv exported.")

 # CLOSE CONNECTION
 
conn.close()

print("\n" + "=" * 60)
print("TABLEAU EXPORT COMPLETED")
print("=" * 60)