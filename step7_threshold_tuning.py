# STEP 7: Threshold tuning for imbalanced classification (GoA is rare)
# -------------------------------------------------------------------
# Why we do this:
# In rare-event prediction, using threshold=0.5 often gives poor F1 because it creates
# too many false positives (or misses positives). Instead, we tune the threshold
# on a validation set and then evaluate on the test set.
#
# This script:
# 1) Loads the prepared train/test data from Step 2
# 2) Splits TRAIN into train/validation
# 3) Trains Logistic Regression and Random Forest
# 4) Searches thresholds and picks the best F1 on validation
# 5) Evaluates that threshold on the TEST set (2023H1)
#
# IMPORTANT: We tune threshold on VALIDATION (not on TEST) to keep evaluation fair.

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# -----------------------------
# SETTINGS (you can change these)
# -----------------------------
# If True, include C40_to_landing_min (same as Step 3)
# If False, remove it (similar idea to Step 4)
USE_TIMING_FEATURE = True

# Threshold grid to search.
# For rare events, useful thresholds are often small (e.g., 0.001 to 0.2).
THRESHOLDS = np.linspace(0.001, 0.2, 200)

RANDOM_STATE = 42

# -----------------------------
# 1) Load data
# -----------------------------
train = pd.read_parquet("data/train_2019_2022_sample.parquet")
test  = pd.read_parquet("data/test_2023H1_full.parquet")

X = train.drop(columns=["GoA"]).copy()
y = train["GoA"].astype(int).copy()

X_test = test.drop(columns=["GoA"]).copy()
y_test = test["GoA"].astype(int).copy()

print("TRAIN:", X.shape, "GoA rate:", y.mean())
print("TEST :", X_test.shape, "GoA rate:", y_test.mean())

# -----------------------------
# 2) Select features
# -----------------------------
base_num_cols = ["C40_BEARING", "C40_CROSS_LAT", "C40_CROSS_LON", "hour", "month", "dayofweek"]
cat_cols = ["airport", "country", "ILS", "market_segment", "AC_CLASS", "AP_C_RWY"]

if USE_TIMING_FEATURE:
    num_cols = ["C40_to_landing_min"] + base_num_cols
else:
    num_cols = base_num_cols

# Ensure numeric columns are numeric
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

# Ensure categorical columns are strings (safe conversion)
def safe_to_str(x):
    try:
        if pd.isna(x):
            return "missing"
    except Exception:
        pass
    return str(x)

for c in cat_cols:
    X[c] = X[c].map(safe_to_str)
    X_test[c] = X_test[c].map(safe_to_str)

# -----------------------------
# 3) Split TRAIN into train/validation
# -----------------------------
# We tune threshold using validation so we don't "cheat" by tuning on test.
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print("\nSplit:")
print("Train-subset:", X_tr.shape, "GoA rate:", y_tr.mean())
print("Val-subset  :", X_val.shape, "GoA rate:", y_val.mean())

# -----------------------------
# 4) Build models (same idea as your step3)
# -----------------------------
# Logistic Regression: scale numeric + one-hot categories
num_pipe_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_lr = ColumnTransformer(
    transformers=[
        ("num", num_pipe_lr, num_cols),
        ("cat", cat_pipe_lr, cat_cols)
    ]
)

model_lr = Pipeline(steps=[
    ("prep", preprocess_lr),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Random Forest: no scaling needed + ordinal encoding
num_pipe_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocess_rf = ColumnTransformer(
    transformers=[
        ("num", num_pipe_rf, num_cols),
        ("cat", cat_pipe_rf, cat_cols)
    ]
)

model_rf = Pipeline(steps=[
    ("prep", preprocess_rf),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

# -----------------------------
# 5) Helper: evaluate at a given threshold
# -----------------------------
def eval_at_threshold(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "threshold": thr,
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "cm": confusion_matrix(y_true, pred)
    }

def find_best_threshold(y_true, proba, thresholds):
    best = None
    for thr in thresholds:
        res = eval_at_threshold(y_true, proba, thr)
        if (best is None) or (res["f1"] > best["f1"]):
            best = res
    return best

# -----------------------------
# 6) Train, tune on validation, test on 2023H1
# -----------------------------
def run_model(name, model):
    print(f"\n=============================")
    print(f"Model: {name}")
    print(f"USE_TIMING_FEATURE = {USE_TIMING_FEATURE}")
    print(f"=============================")

    model.fit(X_tr, y_tr)

    # Validation probabilities
    proba_val = model.predict_proba(X_val)[:, 1]

    # Threshold-independent metrics on validation
    val_roc = roc_auc_score(y_val, proba_val)
    val_pr  = average_precision_score(y_val, proba_val)

    best = find_best_threshold(y_val, proba_val, THRESHOLDS)

    print("\nValidation:")
    print("ROC-AUC:", round(val_roc, 6))
    print("PR-AUC :", round(val_pr, 6))
    print("Best threshold:", round(best["threshold"], 4))
    print("Best F1:", round(best["f1"], 6),
          "Precision:", round(best["precision"], 6),
          "Recall:", round(best["recall"], 6))
    print("Confusion matrix:\n", best["cm"])

    # Test probabilities
    proba_test = model.predict_proba(X_test)[:, 1]

    # Threshold-independent metrics on test
    test_roc = roc_auc_score(y_test, proba_test)
    test_pr  = average_precision_score(y_test, proba_test)

    # Evaluate on test using the best validation threshold
    test_res = eval_at_threshold(y_test, proba_test, best["threshold"])

    print("\nTest (2023H1) using best validation threshold:")
    print("ROC-AUC:", round(test_roc, 6))
    print("PR-AUC :", round(test_pr, 6))
    print("Threshold:", round(best["threshold"], 4))
    print("F1:", round(test_res["f1"], 6),
          "Precision:", round(test_res["precision"], 6),
          "Recall:", round(test_res["recall"], 6))
    print("Confusion matrix:\n", test_res["cm"])

    # Return a summary row for saving
    return {
        "model": name,
        "use_timing_feature": USE_TIMING_FEATURE,
        "val_roc_auc": val_roc,
        "val_pr_auc": val_pr,
        "best_threshold": best["threshold"],
        "val_f1": best["f1"],
        "val_precision": best["precision"],
        "val_recall": best["recall"],
        "test_roc_auc": test_roc,
        "test_pr_auc": test_pr,
        "test_f1": test_res["f1"],
        "test_precision": test_res["precision"],
        "test_recall": test_res["recall"],
    }

rows = []
rows.append(run_model("Logistic Regression", model_lr))
rows.append(run_model("Random Forest", model_rf))

# -----------------------------
# 7) Save summary for report
# -----------------------------
summary = pd.DataFrame(rows)
summary.to_csv("results/threshold_tuning_summary.csv", index=False)
print("\nSaved: results/threshold_tuning_summary.csv")
print("\nSummary:\n", summary)
