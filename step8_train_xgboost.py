# STEP 8: Train and evaluate an XGBoost classifier
# ------------------------------------------------
# This script adds a third model (XGBoost) to compare against Logistic Regression and Random Forest.
# XGBoost is often strong for tabular datasets because it can learn non-linear patterns and interactions.
# We keep the evaluation style consistent with Step 3:
# - ROC-AUC (ranking quality)
# - Precision, Recall, F1 (important for imbalanced classification)
# - Confusion matrix (to see false positives/false negatives clearly)

# Import numpy for numerical operations (arrays, calculations, etc.)
import numpy as np

# Import pandas for loading parquet files and working with tabular data
import pandas as pd

# ColumnTransformer lets us apply different preprocessing steps to numeric vs categorical columns
from sklearn.compose import ColumnTransformer

# Pipeline helps us chain preprocessing + model training in one clean object
from sklearn.pipeline import Pipeline

# OneHotEncoder converts categorical variables into dummy/indicator columns
from sklearn.preprocessing import OneHotEncoder

# SimpleImputer fills missing values (median for numeric, most frequent for categorical)
from sklearn.impute import SimpleImputer

# Import evaluation metrics for classification
from sklearn.metrics import (
    roc_auc_score,         # ROC-AUC measures ranking quality (important for classification)
    precision_score,       # precision = TP / (TP + FP)
    recall_score,          # recall = TP / (TP + FN)
    f1_score,              # harmonic mean of precision and recall
    confusion_matrix       # matrix showing TN, FP, FN, TP
)

# Import XGBoost classifier (must have xgboost installed in the environment)
from xgboost import XGBClassifier


# -----------------------------
# 1) Load prepared data (from Step 2)
# -----------------------------
# The training set is the sampled dataset for 2019–2022
train = pd.read_parquet("data/train_2019_2022_sample.parquet")

# The test set is the real full dataset for 2023 Jan–Jun (no downsampling)
test = pd.read_parquet("data/test_2023H1_full.parquet")

# Separate predictors (X) and target label (y) for training data
X_train = train.drop(columns=["GoA"]).copy()     # drop target column to keep only features
y_train = train["GoA"].astype(int).copy()        # make sure GoA is integer (0/1)

# Separate predictors (X) and target label (y) for test data
X_test = test.drop(columns=["GoA"]).copy()       # drop target column to keep only features
y_test = test["GoA"].astype(int).copy()          # make sure GoA is integer (0/1)

# Print dataset sizes and go-around rates to confirm split and imbalance
print("Train size:", X_train.shape, " GoA rate:", y_train.mean())
print("Test size :", X_test.shape,  " GoA rate:", y_test.mean())


# -----------------------------
# 2) Define feature columns (same idea as Step 3)
# -----------------------------
# Numeric columns: geometry + time variables + timing-to-landing feature
# Note: C40_to_landing_min is included here to match Step 3 behaviour.
num_cols = [
    "C40_BEARING",        # bearing at 40NM crossing
    "C40_CROSS_LAT",      # latitude at 40NM crossing
    "C40_CROSS_LON",      # longitude at 40NM crossing
    "hour",               # time-of-day hour
    "month",              # month of year
    "dayofweek",          # day of week
    "C40_to_landing_min"  # minutes from 40NM crossing to landing time
]

# Categorical columns: operational and aircraft context variables
cat_cols = [
    "airport",           # airport identifier
    "country",           # country of airport
    "ILS",               # ILS identifier for approach/runway
    "market_segment",    # market segment
    "AC_CLASS",          # aircraft class
    "AP_C_RWY"           # runway identifier
]

# Ensure numeric columns are actually numeric.
# If values cannot be converted, they become NaN (and will later be imputed).
for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

# Helper function to safely convert categorical values into strings.
# This prevents errors in encoding when values are missing or not normal strings.
def safe_to_str(x):
    # If value is missing (NaN), return a placeholder category
    try:
        if pd.isna(x):
            return "missing"
    except Exception:
        # If pd.isna fails for some datatype, ignore and convert to string below
        pass

    # Convert everything else to string
    return str(x)

# Apply the string conversion to all categorical columns in train and test.
# This makes one-hot encoding stable and avoids issues with unexpected types.
for c in cat_cols:
    X_train[c] = X_train[c].map(safe_to_str)
    X_test[c] = X_test[c].map(safe_to_str)


# -----------------------------
# 3) Preprocessing: impute + one-hot encode categoricals
# -----------------------------
# We build a preprocessing object that:
# - imputes missing numeric values with the median
# - imputes missing categorical values with the most frequent category
# - one-hot encodes categorical values into dummy variables
preprocess = ColumnTransformer(
    transformers=[
        # Numeric pipeline: replace missing values with median
        ("num", SimpleImputer(strategy="median"), num_cols),

        # Categorical pipeline: fill missing with most common, then one-hot encode
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)


# -----------------------------
# 4) Handle imbalance
# -----------------------------
# In XGBoost, scale_pos_weight can help when the positive class is rare.
# A common rule is: (number of negatives / number of positives) in the training data.
# Important: this is based on the sampled training distribution (not the real-world test rate).
neg = (y_train == 0).sum()        # count how many non-go-arounds
pos = (y_train == 1).sum()        # count how many go-arounds

# Avoid division by zero in case something goes wrong
scale_pos_weight = neg / max(pos, 1)

# Print imbalance ratio so we know what value is being used
print("Training imbalance (neg/pos):", round(scale_pos_weight, 3))


# -----------------------------
# 5) Model: XGBoost classifier
# -----------------------------
# These hyperparameters are chosen as a reasonable starting point:
# - n_estimators: number of trees
# - learning_rate: step size
# - max_depth: complexity of each tree
# - subsample / colsample_bytree: helps reduce overfitting
# - scale_pos_weight: handles imbalance
xgb = XGBClassifier(
    n_estimators=400,            # number of boosting rounds (trees)
    learning_rate=0.05,          # smaller learning rate tends to generalise better
    max_depth=6,                 # depth of each tree (controls complexity)
    subsample=0.8,               # fraction of rows used per tree (regularisation)
    colsample_bytree=0.8,        # fraction of features used per tree (regularisation)
    reg_lambda=1.0,              # L2 regularisation
    min_child_weight=1.0,        # minimum sum hessian needed in a child
    n_jobs=-1,                   # use all CPU cores
    random_state=42,             # reproducibility
    eval_metric="auc",           # evaluation metric inside XGBoost (AUC)
    scale_pos_weight=scale_pos_weight  # imbalance handling
)

# Build a full pipeline: preprocessing first, then the XGBoost model.
# This ensures training and test data receive identical transformations.
model = Pipeline(steps=[
    ("prep", preprocess),
    ("model", xgb)
])

# Train the model on the training set
print("\nTraining XGBoost...")
model.fit(X_train, y_train)


# -----------------------------
# 6) Evaluate on test (threshold = 0.5 like Step 3)
# -----------------------------
# Predict probabilities for the positive class (GoA = 1)
proba = model.predict_proba(X_test)[:, 1]

# Convert probabilities into binary predictions using threshold = 0.5
# This matches the default threshold used in Step 3 to keep comparisons fair.
pred = (proba >= 0.5).astype(int)

# Calculate evaluation metrics
roc = roc_auc_score(y_test, proba)                           # ROC-AUC uses probabilities
prec = precision_score(y_test, pred, zero_division=0)        # precision at threshold
rec = recall_score(y_test, pred, zero_division=0)            # recall at threshold
f1 = f1_score(y_test, pred, zero_division=0)                 # F1 at threshold
cm = confusion_matrix(y_test, pred)                          # confusion matrix

# Store results in a dictionary so it prints neatly (similar to Step 3 output)
results = {
    "ROC_AUC": roc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "ConfusionMatrix": cm
}

# Print results
print("\nXGBoost results:")
print(results)
