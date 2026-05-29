import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report
)

 
# CREATE FOLDERS
 

os.makedirs("models", exist_ok=True)

 
# LOAD WOE DATA
 

print("=" * 60)
print("LOADING WOE DATASETS")
print("=" * 60)

train = pd.read_csv("data/features/train_woe.csv")
test = pd.read_csv("data/features/test_woe.csv")

print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")

 
# SELECT WOE FEATURES
 

woe_cols = [
    c for c in train.columns
    if c.endswith("_woe")
]

print("\nWoE Features Used:")
print(woe_cols)

X_train = train[woe_cols]
y_train = train["target"]

X_test = test[woe_cols]
y_test = test["target"]

 
# TRAIN MODEL
 

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")

 
# PREDICTIONS
 

train_probs = model.predict_proba(X_train)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]

 
# EVALUATION
 

train_auc = roc_auc_score(y_train, train_probs)
test_auc = roc_auc_score(y_test, test_probs)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")

 
# SAVE MODEL
 

with open("models/logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully.")