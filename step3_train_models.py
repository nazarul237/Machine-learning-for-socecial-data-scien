# STEP 3: Train and evaluate models (Logistic Regression + Random Forest)
# ----------------------------------------------------------------------
# In this step, I build two classification models to predict GoA (go-around):
# - Logistic Regression: simple, interpretable baseline
# - Random Forest: non-linear model that can capture interactions
# I use scikit-learn pipelines so preprocessing is consistent and reproducible.
# Core data libraries.
import pandas as pd
import numpy as np

# scikit-learn tools for building clean, repeatable ML pipelines.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Models used in this comparison.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics (important for rare-event classification like go-arounds).
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score
)

# -----------------------------
# 1) Load the prepared train/test files (fast)
# -----------------------------
# Load the prepared datasets from Step 2.
# TRAIN is sampled (higher GoA rate); TEST is realistic (very low GoA rate).
train = pd.read_parquet("data/train_2019_2022_sample.parquet")
test  = pd.read_parquet("data/test_2023H1_full.parquet")

# Separate predictors (X) from the target label (y).
X_train = train.drop(columns=["GoA"]).copy()
y_train = train["GoA"].astype(int).copy()

X_test  = test.drop(columns=["GoA"]).copy()
y_test  = test["GoA"].astype(int).copy()

# Quick check: dataset sizes and class imbalance.
print("Train size:", X_train.shape, " GoA rate:", y_train.mean())
print("Test size :", X_test.shape,  " GoA rate:", y_test.mean())

# -----------------------------
# 2) Column lists (MUST match your saved parquet columns)
# -----------------------------
# Numeric features: 40NM geometry + time features.
# NOTE: This list includes C40_to_landing_min (checked for leakage in Step 4).
num_cols = [
    "C40_to_landing_min", "C40_BEARING", "C40_CROSS_LAT", "C40_CROSS_LON",
    "hour", "month", "dayofweek"
]
# Categorical features: airport/runway/ILS and aircraft segment/class information.
cat_cols = ["airport", "country", "ILS", "market_segment", "AC_CLASS", "AP_C_RWY"]

# -----------------------------
# 3) Clean data types (THIS fixes your error)
#    - Numeric columns must be numbers
#    - Categorical columns must be simple strings (no arrays/lists)
# -----------------------------
# Ensure numeric columns are numeric (anything invalid becomes NaN and will be imputed).
for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
    X_test[c]  = pd.to_numeric(X_test[c], errors="coerce")

# Helper to safely convert categorical values to strings.
# Some cells can be arrays/lists; this avoids encoder errors.
def safe_to_str(x):
    # pd.isna(x) can crash if x is an array/list -> we catch it
    try:
        if pd.isna(x):
            return "missing"
    except Exception:
        # if it's a list/array/etc, treat it as not-missing
        pass
    return str(x)

# Convert all categorical columns to strings and handle missing values consistently.
for c in cat_cols:
    X_train[c] = X_train[c].map(safe_to_str)
    X_test[c]  = X_test[c].map(safe_to_str)

# Safety check: if a required column is missing, stop early with a clear error.
# Safety check (so we don't get "column not found" errors)
missing_num = [c for c in num_cols if c not in X_train.columns]
missing_cat = [c for c in cat_cols if c not in X_train.columns]
print("\nMissing num cols:", missing_num)
print("Missing cat cols:", missing_cat)

if missing_num or missing_cat:
    raise ValueError("Some columns are missing. Check your train/test parquet files.")

# -----------------------------
# 4) Model 1: Logistic Regression (one-hot for categories)
# -----------------------------
# Logistic Regression preprocessing:
# - impute missing numeric values (median)
# - scale numeric features (LR is sensitive to feature scaling)
num_pipe_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# One-hot encoding for categories: creates dummy variables for each category level.
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

# Fit Logistic Regression with class_weight='balanced' to account for imbalance in TRAIN.
model_lr = Pipeline(steps=[
    ("prep", preprocess_lr),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# -----------------------------
# 5) Model 2: Random Forest (OrdinalEncoder is simpler + faster)
# -----------------------------
# Random Forest preprocessing: impute missing values (no scaling required).
num_pipe_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# OrdinalEncoder maps categories to integer codes (fast for tree models).
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

# Random Forest model (200 trees). random_state makes results reproducible.
model_rf = Pipeline(steps=[
    ("prep", preprocess_rf),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

# -----------------------------
# 6) Evaluation function
# -----------------------------
# Evaluation function: uses probability outputs for ROC-AUC and a 0.5 threshold for class labels.
def evaluate(model, X_te, y_te, threshold=0.5):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)

    return {
        "ROC_AUC": roc_auc_score(y_te, proba),
        "F1": f1_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "ConfusionMatrix": confusion_matrix(y_te, pred)
    }

# -----------------------------
# 7) Train + test both models
# -----------------------------
# Train and evaluate both models on the holdout test set (2023H1).
print("\nTraining Logistic Regression...")
model_lr.fit(X_train, y_train)
res_lr = evaluate(model_lr, X_test, y_test)
print("Logistic Regression results:")
print(res_lr)

print("\nTraining Random Forest...")
model_rf.fit(X_train, y_train)
res_rf = evaluate(model_rf, X_test, y_test)
print("Random Forest results:")
print(res_rf)
