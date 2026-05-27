import pandas as pd
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)

# =====================================================
# CREATE FOLDERS
# =====================================================

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("LOADING MODEL AND DATA")
print("=" * 60)

with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

train_woe = pd.read_csv("data/features/train_woe.csv")
test_woe = pd.read_csv("data/features/test_woe.csv")

woe_cols = [
    c for c in train_woe.columns
    if c.endswith("_woe")
]

print(f"WoE Features: {len(woe_cols)}")

# =====================================================
# PREDICT PROBABILITIES
# =====================================================

print("\nGenerating predictions...")

y_true = test_woe["target"]

y_prob = model.predict_proba(
    test_woe[woe_cols]
)[:, 1]

# =====================================================
# GINI
# =====================================================

auc = roc_auc_score(y_true, y_prob)

gini = 2 * auc - 1

print("\n" + "=" * 60)
print("MODEL VALIDATION")
print("=" * 60)

print(f"AUC:  {auc:.4f}")
print(f"Gini: {gini:.4f}")

# =====================================================
# KS STATISTIC
# =====================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

ks = np.max(tpr - fpr)

print(f"KS:   {ks:.4f}")

# =====================================================
# ROC + KS PLOT
# =====================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

# ROC Curve
ax1.plot(
    fpr,
    tpr,
    label=f"AUC={auc:.3f}\nGini={gini:.3f}"
)

ax1.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax1.set_title("ROC Curve")

ax1.set_xlabel("False Positive Rate")

ax1.set_ylabel("True Positive Rate")

ax1.legend()

# KS Chart
idx = np.argmax(tpr - fpr)

ax2.plot(
    thresholds[1:],
    tpr[1:],
    label="TPR"
)

ax2.plot(
    thresholds[1:],
    fpr[1:],
    label="FPR"
)

ax2.axvline(
    thresholds[idx],
    linestyle="--",
    label=f"KS={ks:.3f}"
)

ax2.set_title("KS Statistic")

ax2.set_xlabel("Threshold")

ax2.legend()

plt.tight_layout()

plt.savefig(
    "outputs/plots/roc_ks_chart.png",
    dpi=150
)

plt.close()

print("\nROC + KS plot saved.")

# =====================================================
# PSI FUNCTION
# =====================================================

def psi(expected, actual, bins=10):

    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )

    expected_pct = (
        np.histogram(expected, breakpoints)[0]
        / len(expected)
    )

    actual_pct = (
        np.histogram(actual, breakpoints)[0]
        / len(actual)
    )

    expected_pct = np.where(
        expected_pct == 0,
        0.0001,
        expected_pct
    )

    actual_pct = np.where(
        actual_pct == 0,
        0.0001,
        actual_pct
    )

    psi_value = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return psi_value

# =====================================================
# PSI CALCULATION
# =====================================================

print("\nCalculating PSI...")

half = len(test_woe) // 2

score_ref = model.predict_proba(
    test_woe.iloc[:half][woe_cols]
)[:, 1]

score_monitor = model.predict_proba(
    test_woe.iloc[half:][woe_cols]
)[:, 1]

psi_value = psi(
    score_ref,
    score_monitor
)

print(f"PSI:  {psi_value:.4f}")

# =====================================================
# PSI STATUS
# =====================================================

if psi_value < 0.1:
    psi_status = "STABLE"

elif psi_value < 0.2:
    psi_status = "MINOR SHIFT"

else:
    psi_status = "MAJOR SHIFT"

print(f"Status: {psi_status}")

# =====================================================
# VALIDATION REPORT
# =====================================================

report = pd.DataFrame({
    "Metric": ["AUC", "Gini", "KS", "PSI"],
    "Value": [
        round(auc, 4),
        round(gini, 4),
        round(ks, 4),
        round(psi_value, 4)
    ],
    "Target": [
        ">0.70",
        ">0.40",
        ">0.30",
        "<0.10"
    ],
    "Status": [
        "Pass" if auc > 0.70 else "Fail",
        "Pass" if gini > 0.40 else "Fail",
        "Pass" if ks > 0.30 else "Fail",
        "Pass" if psi_value < 0.10 else "Fail"
    ]
})

report.to_csv(
    "outputs/reports/validation_report.csv",
    index=False
)

print("\nValidation report saved.")

print("\n")
print(report)