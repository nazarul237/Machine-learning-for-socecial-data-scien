"""
Step 9 — ROC Curves (LR vs RF vs XGBoost) on the 2023H1 Test Set

In this script, I reproduce the single ROC figure used in my report by:
1) Loading the prepared TRAIN and TEST parquet files created in Step 2.
2) Training three classification models (Logistic Regression, Random Forest, XGBoost).
3) Predicting probabilities on the 2023H1 test set.
4) Plotting all three ROC curves on one graph and saving the figure.

Output:
- results/roc_curves_2023H1.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

from xgboost import XGBClassifier


# ============================================================
# 1) Load the prepared datasets (created earlier in Step 2)
# ============================================================
# I use the sampled training set (2019–2022) and the realistic test set (2023H1).
train = pd.read_parquet("data/train_2019_2022_sample.parquet")
test = pd.read_parquet("data/test_2023H1_full.parquet")

# Separate predictors (X) and target label (y).
# GoA is the binary outcome: 1 = go-around, 0 = normal landing.
X_train = train.drop(columns=["GoA"]).copy()
y_train = train["GoA"].astype(int).copy()

X_test = test.drop(columns=["GoA"]).copy()
y_test = test["GoA"].astype(int).copy()

print("Train size:", X_train.shape, "GoA rate:", round(y_train.mean(), 6))
print("Test size :", X_test.shape, "GoA rate:", round(y_test.mean(), 6))


# ============================================================
# 2) Define the features used in modelling
# ============================================================
# These are the same categories used in my earlier model training scripts.
# Numeric features include geometry and time features plus the timing-derived feature.
num_cols = [
    "C40_to_landing_min",
    "C40_BEARING",
    "C40_CROSS_LAT",
    "C40_CROSS_LON",
    "hour",
    "month",
    "dayofweek"
]

# Categorical features describe operational and aircraft context.
cat_cols = ["airport", "country", "ILS", "market_segment", "AC_CLASS", "AP_C_RWY"]


# ============================================================
# 3) Minimal cleaning to ensure consistent data types
# ============================================================
# I force numeric columns to be numeric; invalid entries become NaN and will be imputed.
for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

# I convert categorical columns to strings so encoders behave consistently.
# Missing values are replaced with a clear token.
def safe_to_str(x):
    try:
        if pd.isna(x):
            return "missing"
    except Exception:
        pass
    return str(x)

for c in cat_cols:
    X_train[c] = X_train[c].map(safe_to_str)
    X_test[c] = X_test[c].map(safe_to_str)


# ============================================================
# 4) Build the three model pipelines (preprocessing + model)
# ============================================================

# ---- Logistic Regression (LR) ----
# For LR, I one-hot encode categoricals and scale numeric variables.
# I also use class_weight="balanced" due to strong class imbalance.
lr_preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

lr_pipeline = Pipeline([
    ("prep", lr_preprocess),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# ---- Random Forest (RF) ----
# For tree-based RF, scaling is not required.
# I use ordinal encoding for categoricals to keep the feature space compact.
rf_preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]), cat_cols)
])

rf_pipeline = Pipeline([
    ("prep", rf_preprocess),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

# ---- XGBoost (XGB) ----
# XGBoost uses scale_pos_weight to account for imbalance (neg/pos ratio in training data).
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
scale_pos_weight = neg / max(pos, 1)

xgb_preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

xgb_pipeline = Pipeline([
    ("prep", xgb_preprocess),
    ("model", XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist"
    ))
])


# ============================================================
# 5) Train each model and compute ROC curve on 2023H1 test set
# ============================================================
models = [
    ("Logistic Regression", lr_pipeline),
    ("Random Forest", rf_pipeline),
    ("XGBoost", xgb_pipeline)
]

roc_results = []  # I store each model's fpr, tpr and auc for plotting.

for name, pipeline in models:
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)

    # I use predicted probabilities for the positive class (GoA = 1),
    # because ROC curves are built from probability scores rather than hard labels.
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ROC-AUC measures ranking quality and is useful for imbalanced classification.
    auc = roc_auc_score(y_test, y_prob)

    # roc_curve returns the false positive rate and true positive rate at all thresholds.
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_results.append((name, fpr, tpr, auc))
    print(f"{name} ROC-AUC: {auc:.3f}")


# ============================================================
# 6) Plot all ROC curves on a single figure and save it
# ============================================================
# I save the figure into the results/ folder so it can be inserted into the report.
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 6))

for name, fpr, tpr, auc in roc_results:
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

# Baseline reference for a random classifier
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.title("ROC Curves on 2023H1 Test Set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()

out_path = "results/roc_curves_2023H1.png"
plt.savefig(out_path, dpi=200)
plt.show()

print(f"\nSaved ROC figure to: {out_path}")