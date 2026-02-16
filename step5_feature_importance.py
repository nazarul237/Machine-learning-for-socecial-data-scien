# STEP 5: Feature importance for the final Logistic Regression model
# ------------------------------------------------------------------
# Logistic Regression gives a coefficient per feature (interpretable).
# Positive coef increases GoA probability; negative coef decreases it.
# I export a full coefficient table so I can discuss key drivers in the report.
# Core data libraries.
import pandas as pd
import numpy as np
from pathlib import Path

# scikit-learn pipeline tools.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Load prepared train/test files
# -----------------------------
# Load training data (sampled).
train = pd.read_parquet("data/train_2019_2022_sample.parquet")
# Load test data (not strictly required here, but kept for consistency).
test = pd.read_parquet("data/test_2023H1_full.parquet")  # not required for coefficients, but fine

# Split predictors (X) from the target label (y).
X_train = train.drop(columns=["GoA"]).copy()
y_train = train["GoA"].astype(int).copy()


# -----------------------------
# FINAL MODEL FEATURES (leakage-check version)
# IMPORTANT: no C40_to_landing_min here
# -----------------------------
# Final feature set (leakage-safe): C40_to_landing_min is not included.
num_cols = [
    "C40_BEARING", "C40_CROSS_LAT", "C40_CROSS_LON",
    "hour", "month", "dayofweek"
]
cat_cols = ["airport", "country", "ILS", "market_segment", "AC_CLASS", "AP_C_RWY"]


# -----------------------------
# Clean data types (prevents weird errors)
# -----------------------------
# Convert numeric columns to numeric types so the pipeline can impute correctly.
for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")

# Convert categorical values to strings; map missing values to 'missing' to avoid encoder issues.
def safe_to_str(x):
    try:
        if pd.isna(x):
            return "missing"
    except Exception:
        pass
    return str(x)

for c in cat_cols:
    X_train[c] = X_train[c].map(safe_to_str)


# -----------------------------
# Build Logistic Regression pipeline
# -----------------------------
# Numeric preprocessing: impute missing values and scale (helps LR).
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing: impute and one-hot encode; ignore unseen categories in test.
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ]
)

# Fit Logistic Regression with balanced class weights (handles imbalance).
model_lr = Pipeline(steps=[
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

print("Training final Logistic Regression (for feature importance)...")
model_lr.fit(X_train, y_train)
print("Done.\n")


# -----------------------------
# Extract feature names + coefficients
# -----------------------------
# Extract the fitted preprocessing step and classifier so I can pull feature names + coefficients.
prep = model_lr.named_steps["prep"]
clf = model_lr.named_steps["model"]

# Get feature names created by ColumnTransformer + OneHotEncoder
# After one-hot encoding, feature_names includes expanded dummy variables (one per category level).
feature_names = prep.get_feature_names_out()
coefs = clf.coef_.ravel()

# Build a tidy coefficient table and sort by abs_coef to see most influential features first.
imp = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
   "abs_coef": np.abs(coefs)
}).sort_values("abs_coef", ascending=False)

# Show top features overall + top positive/negative
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 140)

print("Top 15 most important features (by absolute coefficient):")
print(imp.head(15)[["feature", "coef"]])

print("\nTop 10 features that INCREASE go-around probability (positive coef):")
print(imp[imp["coef"] > 0].head(10)[["feature", "coef"]])

print("\nTop 10 features that DECREASE go-around probability (negative coef):")
print(imp[imp["coef"] < 0].head(10)[["feature", "coef"]])

# Save coefficients for later analysis and for including summary tables in the report.
# Save for your report
Path("results").mkdir(exist_ok=True)
imp.to_csv("results/logreg_feature_importance.csv", index=False)
print("\nSaved: results/logreg_feature_importance.csv")
