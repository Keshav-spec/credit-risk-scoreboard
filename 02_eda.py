import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots folder
os.makedirs("outputs/plots", exist_ok=True)

# Load cleaned data
df = pd.read_csv("data/processed/loans_clean.csv")

# Style
sns.set_style("whitegrid")

 
# 1. Default Rate by Grade
 

grade_default = (
    df.groupby("grade")["target"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=grade_default,
    x="grade",
    y="target"
)

plt.title("Default Rate by Loan Grade")
plt.ylabel("Default Rate")

plt.savefig(
    "outputs/plots/default_by_grade.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

 
# 2. Loan Amount Distribution
 

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="loan_amnt",
    hue="target",
    bins=40,
    alpha=0.6
)

plt.title("Loan Amount Distribution")

plt.savefig(
    "outputs/plots/loan_amnt_dist.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

 
# 3. FICO Score Distribution
 

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="target",
    y="fico_range_low"
)

plt.title("FICO Score vs Default")

plt.savefig(
    "outputs/plots/fico_boxplot.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

 
# 4. Correlation Heatmap
 

plt.figure(figsize=(12, 10))

corr = df.select_dtypes(include=np.number).corr()

sns.heatmap(
    corr,
    cmap="RdYlGn",
    center=0,
    linewidths=0.3
)

plt.title("Feature Correlation Matrix")

plt.savefig(
    "outputs/plots/correlation_heatmap.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("EDA plots saved successfully.")