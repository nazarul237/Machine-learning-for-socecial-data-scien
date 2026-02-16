# STEP 4: Leakage / robustness check
# ----------------------------------
# Here I test whether my results depend on a potentially 'leaky' feature.
# C40_to_landing_min uses landing_time, which happens after the go-around decision.
# So I remove it and re-run the same models to check if performance stays similar.
# Core data libraries.
import pandas as pd
import numpy as np

# scikit-learn tools for preprocessing and pipelines.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
# Load the same prepared datasets as Step 3 (fair comparison).
train = pd.read_parquet("data/train_2019_2022_sample.parquet")
test  = pd.read_parquet("data/test_2023H1_full.parquet")

X_train = train.drop(columns=["GoA"]).copy()
y_train = train["GoA"].astype(int).copy()

X_test  = test.drop(columns=["GoA"]).copy()
y_test  = test["GoA"].astype(int).copy()

# Check sizes and GoA rates (class imbalance is large in TEST).
print("Train size:", X_train.shape, " GoA rate:", y_train.mean())
print("Test size :", X_test.shape,  " GoA rate:", y_test.mean())

# -----------------------------
# 2) Column lists (MUST match your saved parquet columns)
# -----------------------------
# Numeric features used in the leakage-check run.
# IMPORTANT: C40_to_landing_min is excluded here.
num_cols = [
    "C40_BEARING", "C40_CROSS_LAT", "C40_CROSS_LON",
    "hour", "month", "dayofweek"
]

cat_cols = ["airport", "country", "ILS", "market_segment", "AC_CLASS", "AP_C_RWY"]

# -----------------------------
# 3) Clean data types (THIS fixes your error)
#    - Numeric columns must be numbers
#    - Categorical columns must be simple strings (no arrays/lists)
# -----------------------------
# Convert numeric columns to numeric types; invalid values become NaN for the imputer.
for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
    X_test[c]  = pd.to_numeric(X_test[c], errors="coerce")

# Convert categorical values to strings and handle missing values safely (same as Step 3).
def safe_to_str(x):
    # pd.isna(x) can crash if x is an array/list -> we catch it
    try:
        if pd.isna(x):
            return "missing"
    except Exception:
        # if it's a list/array/etc, treat it as not-missing
        pass
    return str(x)

for c in cat_cols:
    X_train[c] = X_train[c].map(safe_to_str)
    X_test[c]  = X_test[c].map(safe_to_str)

# Safety check for missing columns.
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
# Logistic Regression preprocessing: impute + scale.
num_pipe_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# One-hot encoding for Logistic Regression.
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

# Logistic Regression model with balanced class weights.
model_lr = Pipeline(steps=[
    ("prep", preprocess_lr),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# -----------------------------
# 5) Model 2: Random Forest (OrdinalEncoder is simpler + faster)
# -----------------------------
# Random Forest preprocessing: impute only (no scaling needed).
num_pipe_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Ordinal encoding for Random Forest (fast).
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
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

# -----------------------------
# 6) Evaluation function
# -----------------------------
# Same evaluation function as Step 3 so results are directly comparable.
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
# Train and evaluate models without the timing feature.
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
