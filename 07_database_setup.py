import sqlite3
import pandas as pd
import os

#  
# CREATE DATABASE FOLDER
#  

os.makedirs("data", exist_ok=True)

#  
# LOAD SCORED DATA
#  

print("=" * 60)
print("LOADING SCORED DATA")
print("=" * 60)

df = pd.read_csv(
    "data/features/test_scored.csv"
)

print(f"Dataset Shape: {df.shape}")

#  
# CREATE SQLITE DATABASE
#  

db_path = "data/credit_risk.db"

conn = sqlite3.connect(db_path)

print("\nCreating database table...")

df.to_sql(
    "loan_scores",
    conn,
    if_exists="replace",
    index=False
)

print("Table created successfully.")

#  
# ADD SCORE BAND COLUMN
#  

print("\nCreating score bands...")

conn.execute("""
ALTER TABLE loan_scores
ADD COLUMN score_band TEXT
""")

conn.execute("""
UPDATE loan_scores
SET score_band =
CASE
    WHEN credit_score < 400 THEN 'Very High Risk'
    WHEN credit_score < 500 THEN 'High Risk'
    WHEN credit_score < 600 THEN 'Medium Risk'
    WHEN credit_score < 700 THEN 'Low Risk'
    ELSE 'Very Low Risk'
END
""")

conn.commit()

#  
# VALIDATION
#  

summary = pd.read_sql("""
SELECT score_band,
       COUNT(*) AS count
FROM loan_scores
GROUP BY score_band
""", conn)

print("\nScore Band Distribution:")
print(summary)

conn.close()

print("\n" + "=" * 60)
print("DATABASE SETUP COMPLETED")
print("=" * 60)

print(f"Database saved at: {db_path}")