import pandas as pd
import numpy as np
import pickle
import os

 
# CREATE OUTPUT FOLDER
 

os.makedirs("data/features", exist_ok=True)

 
# LOAD MODEL + DATA
 

print("=" * 60)
print("SIMULATING MONTHLY MODEL DRIFT")
print("=" * 60)

with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

test_woe = pd.read_csv(
    "data/features/test_woe.csv"
)

woe_cols = [
    c for c in test_woe.columns
    if c.endswith("_woe")
]

monthly_data = []

 
# SIMULATE 6 MONTHS
 

for month in range(1, 7):

    print(f"\nGenerating month 2025-{month:02d}")

    # Progressive drift
    drift_strength = 0.02 * month

    noise = np.random.normal(
        0,
        drift_strength,
        size=test_woe[woe_cols].shape
    )

    X_noisy = (
        test_woe[woe_cols].values
        + noise
    )

    # Predict probabilities
    probs = model.predict_proba(X_noisy)[:, 1]

    # Convert to score
    odds = probs / (1 - probs + 1e-9)

    scores = (
        600 - 50 * np.log(odds)
    ).clip(300, 850)

    month_df = pd.DataFrame({
        "month": f"2025-{month:02d}",
        "credit_score": scores.round(0),
        "prob_default": probs,
        "target": test_woe["target"].values
    })

    # Score bands
    month_df["score_band"] = pd.cut(
        month_df["credit_score"],
        bins=[0, 400, 500, 600, 700, 850],
        labels=[
            "Very High Risk",
            "High Risk",
            "Medium Risk",
            "Low Risk",
            "Very Low Risk"
        ]
    )

    monthly_data.append(month_df)

 
# COMBINE ALL MONTHS
 

monthly_df = pd.concat(
    monthly_data,
    ignore_index=True
)

 
# SAVE FILE
 

monthly_df.to_csv(
    "data/features/monthly_scores.csv",
    index=False
)

print("\nMonthly simulation complete.")

print(f"\nFinal Shape: {monthly_df.shape}")

print("\nSaved:")
print("data/features/monthly_scores.csv")