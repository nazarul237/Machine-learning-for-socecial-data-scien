import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Load prepared train/test files
# -----------------------------
train = pd.read_parquet("data/train_2019_2022_sample.parquet")
test = pd.read_parquet("data/test_2023H1_full.parquet")  # not required for coefficients, but fine

X_train = train.drop(columns=["GoA"]).copy()
y_train = train["GoA"].astype(int).copy()


# -----------------------------
# FINAL MODEL FEATURES (leakage-check version)
# IMPORTANT: no C40_to_landing_min here
# -----------------------------
num_cols = [
    "C40_BEARING", "C40_CROSS_LAT", "C40_CROSS_LON",
    "hour", "month", "dayofweek"
]
cat_cols = ["airport", "country", "ILS", "market_segment", "AC_CLASS", "AP_C_RWY"]


# -----------------------------
# Clean data types (prevents weird errors)
# -----------------------------
for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")

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
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

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
prep = model_lr.named_steps["prep"]
clf = model_lr.named_steps["model"]

# Get feature names created by ColumnTransformer + OneHotEncoder
feature_names = prep.get_feature_names_out()
coefs = clf.coef_.ravel()

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

# Save for your report
Path("results").mkdir(exist_ok=True)
imp.to_csv("results/logreg_feature_importance.csv", index=False)
print("\nSaved: results/logreg_feature_importance.csv")