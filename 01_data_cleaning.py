import pandas as pd
import numpy as np
import os

# Create folders
os.makedirs("data/processed", exist_ok=True)

# Load dataset
df = pd.read_csv(
    "data/raw/loan_sample_100k.csv",
    low_memory=False
)

print("=" * 50)
print("INITIAL DATA")
print("=" * 50)
print("Shape:", df.shape)
print("\nFirst 10 columns:")
print(df.columns[:10].tolist())

# Missing value analysis
missing_pct = df.isnull().mean().sort_values(ascending=False)

# Drop columns with >40% missing values
drop_cols = missing_pct[missing_pct > 0.4].index.tolist()

df.drop(columns=drop_cols, inplace=True)

print(f"\nDropped {len(drop_cols)} high-missing columns")

print("\nDataset shape after missing-value filtering:")
print(df.shape)

# TARGET VARIABLE CREATION

TARGET_MAP = {
    "Fully Paid": 0,
    "Current": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Late (16-30 days)": 1,
    "In Grace Period": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
}

# Keep only required statuses
df = df[df["loan_status"].isin(TARGET_MAP.keys())]

# Create target
df["target"] = df["loan_status"].map(TARGET_MAP)

print("\nTARGET DISTRIBUTION")
print(df["target"].value_counts())

default_rate = df["target"].mean()

print(f"\nDefault Rate: {default_rate:.4f}")

# Convert percentage columns if needed
percent_cols = ["int_rate", "revol_util"]

for col in percent_cols:
    if df[col].dtype == "object":
        df[col] = (
            df[col]
            .str.replace("%", "", regex=False)
            .astype(float)
        )

# FEATURE SELECTION

FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type"
]

# Keep selected columns only
df = df[FEATURES + ["target"]]

# Drop remaining null rows
df.dropna(inplace=True)

print("\nFINAL CLEAN DATASET")
print(df.shape)

# Save cleaned dataset
df.to_csv(
    "data/processed/loans_clean.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")