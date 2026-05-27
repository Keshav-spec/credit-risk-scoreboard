import pandas as pd
import scorecardpy as sc
import pickle
import os

from sklearn.model_selection import train_test_split

# =====================================================
# CREATE FOLDERS
# =====================================================

os.makedirs("outputs/reports", exist_ok=True)

# =====================================================
# LOAD MODEL + BINS
# =====================================================

print("=" * 60)
print("LOADING MODEL AND WOE BINS")
print("=" * 60)

with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/woe_bins.pkl", "rb") as f:
    bins = pickle.load(f)

# =====================================================
# EXTRACT ORIGINAL FEATURE NAMES
# =====================================================

xcolumns = [
    c.replace("_woe", "")
    for c in model.feature_names_in_
]

print("\nFeatures used in scorecard:")
print(xcolumns)

# =====================================================
# GENERATE SCORECARD
# =====================================================

print("\nGenerating scorecard...")

card = sc.scorecard(
    bins,
    model,
    xcolumns=xcolumns,
    points0=600,
    odds0=1/20,
    pdo=50
)

print("Scorecard generated successfully.")

# =====================================================
# SAVE SCORECARD TABLE
# =====================================================

scorecard_df = pd.concat(card.values())

scorecard_df.to_csv(
    "outputs/reports/scorecard_table.csv",
    index=False
)

print("\nScorecard table saved.")

print(scorecard_df.head())

# =====================================================
# SCORE TEST DATA
# =====================================================

print("\nScoring test dataset...")

test_raw = pd.read_csv(
    "data/processed/loans_clean.csv"
)

_, test_raw_split = train_test_split(
    test_raw,
    test_size=0.3,
    random_state=42,
    stratify=test_raw["target"]
)

test_raw_split["credit_score"] = sc.scorecard_ply(
    test_raw_split,
    card,
    print_step=0
)["score"]

# =====================================================
# SAVE SCORED DATA
# =====================================================

test_raw_split.to_csv(
    "data/features/test_scored.csv",
    index=False
)

print("\nScored dataset saved.")

print(
    f"\nScore Range: "
    f"{test_raw_split['credit_score'].min():.0f}"
    f" - "
    f"{test_raw_split['credit_score'].max():.0f}"
)