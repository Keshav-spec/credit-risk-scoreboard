import pandas as pd
import scorecardpy as sc
import os
import pickle

from sklearn.model_selection import train_test_split

# CREATE FOLDERS

os.makedirs("data/features", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

 
# LOAD DATA
 

print("=" * 60)
print("LOADING CLEAN DATASET")
print("=" * 60)

df = pd.read_csv("data/processed/loans_clean.csv")

print(f"Dataset Shape: {df.shape}")

 
# TRAIN TEST SPLIT
 

print("\nSplitting train/test datasets...")

train, test = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["target"]
)

print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")

 
# WOE BINNING
 

print("\nPerforming WoE binning...")

bins = sc.woebin(
    train,
    y="target",
    method="tree",
    bin_num_limit=8
)

print("WoE binning completed.")

 
# INFORMATION VALUE
 

print("\nCalculating Information Value (IV)...")

iv_df = sc.iv(
    train,
    y="target"
)

iv_df = iv_df.sort_values(
    "info_value",
    ascending=False
)

print("\nTop Features by IV:")
print(iv_df.head(10))

# Save IV report
iv_df.to_csv(
    "outputs/reports/iv_table.csv",
    index=False
)

 
# FEATURE SELECTION
 

good_features = iv_df[
    iv_df["info_value"] >= 0.1
]["variable"].tolist()

print("\nSelected Features:")
print(good_features)

print(f"\nTotal Selected Features: {len(good_features)}")

 
# FILTER SELECTED BINS
 

bins_selected = {
    k: bins[k]
    for k in good_features
    if k in bins
}

 
# APPLY WOE TRANSFORMATION
 

print("\nApplying WoE transformation...")

train_woe = sc.woebin_ply(
    train[good_features + ["target"]],
    bins_selected
)

test_woe = sc.woebin_ply(
    test[good_features + ["target"]],
    bins_selected
)

print("WoE transformation complete.")

 
# SAVE TRANSFORMED DATASETS
 

train_woe.to_csv(
    "data/features/train_woe.csv",
    index=False
)

test_woe.to_csv(
    "data/features/test_woe.csv",
    index=False
)

print("\nWoE datasets saved.")

 
# SAVE BINS
 

with open("models/woe_bins.pkl", "wb") as f:
    pickle.dump(bins_selected, f)

print("WoE bins saved.")

 
# FINAL SUMMARY
 

print("\n" + "=" * 60)
print("WOE ENCODING COMPLETED")
print("=" * 60)

print(f"Train WoE Shape: {train_woe.shape}")
print(f"Test WoE Shape: {test_woe.shape}")