# STEP 2: Prepare train/test datasets + basic feature engineering
# --------------------------------------------------------------
# What I do in this step:
# 1) Read the large raw dataset efficiently (PyArrow).
# 2) Build a training set from 2019–2022 with negative downsampling (to make modelling feasible).
# 3) Build a realistic holdout test set from 2023 Jan–Jun (no downsampling).
# 4) Create a small set of simple features (time-of-day + 40NM crossing features).
# 5) Save the prepared datasets as parquet so later scripts run quickly.
# Import the main libraries used for data handling and efficient parquet filtering.
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from datetime import datetime, timezone
import gc

# Path to the raw dataset file (large).
# I keep it as a single variable so it is easy to change later if needed.
path = "data/osn23_landings_merged.parquet.gzip"

# I restrict to 'safe' predictors: airport/runway/ILS, aircraft segment/class,
# and 40NM crossing geometry. I avoid columns that could 'cheat' by directly
# encoding the outcome (e.g., attempt counters/timestamps that imply a go-around).
# Safe columns (no cheating with attempt count/times)
keep_cols = [
    "landing_time", "airport", "country",
    "ILS", "market_segment", "AC_CLASS",
    "AP_C_RWY",
    "C40_CROSS_TIME", "C40_BEARING", "C40_CROSS_LAT", "C40_CROSS_LON",
    "GoA"
]

# NEG_RATIO controls the negative downsampling in TRAIN.
# Example: NEG_RATIO=20 means I keep up to 20 non-go-arounds for every go-around.
NEG_RATIO = 20
RANDOM_STATE = 42

# Create a PyArrow Dataset object so I can filter by landing_time without loading everything into memory.
dataset = ds.dataset(path, format="parquet")

# Helper function: convert Python datetime into a PyArrow timestamp (UTC).
# This makes the time filters match the parquet timestamp type.
def arrow_ts(dt):
    # Make a timestamp that matches the dataset type (UTC timestamp)
    return pa.scalar(dt, type=pa.timestamp("ns", tz="UTC"))

# I collect each year's sampled training data into this list, then concatenate at the end.
train_parts = []

# --------
# TRAIN = 2019–2022 (sample negatives, keep all go-arounds)
# --------
# Loop year-by-year to keep memory usage controlled.
# For each year: load rows in that year, keep all positives, sample negatives, then store the result.
for year in [2019, 2020, 2021, 2022]:
    print(f"\nLoading year {year}...")

    # Define the start/end timestamps for the current year (UTC).
    start = arrow_ts(datetime(year, 1, 1, tzinfo=timezone.utc))
    # Define the start/end timestamps for the current year (UTC).
    end   = arrow_ts(datetime(year + 1, 1, 1, tzinfo=timezone.utc))

    # Filter expression: only keep rows whose landing_time is within this year.
    filt = (ds.field("landing_time") >= start) & (ds.field("landing_time") < end)

    # Read the filtered subset and only the columns I need (faster + less memory).
    table = dataset.to_table(columns=keep_cols, filter=filt)
    # Convert the PyArrow table into a pandas DataFrame for easier manipulation.
    df_year = table.to_pandas()

    # Target 0/1
    # Ensure the target label is stored as integer 0/1.
    df_year["GoA"] = df_year["GoA"].astype(int)

    # Split into positive class (go-around) and negative class (normal landing).
    pos = df_year[df_year["GoA"] == 1]
    neg = df_year[df_year["GoA"] == 0]

    # Sample negatives to reduce class imbalance and keep the training set manageable.
    n_neg = min(len(neg), len(pos) * NEG_RATIO)
    neg_sample = neg.sample(n=n_neg, random_state=RANDOM_STATE)

    # Combine positives + sampled negatives, then shuffle to mix classes.
    df_small = pd.concat([pos, neg_sample]).sample(frac=1, random_state=RANDOM_STATE)

    print(f"Year {year}: total={len(df_year):,}  pos={len(pos):,}  neg_sample={len(neg_sample):,}  kept={len(df_small):,}")

    # Add this year's sampled data to the list so I can concatenate later.
    train_parts.append(df_small)

    # Free memory (important!)
    # Explicitly delete large objects and run garbage collection (important for big datasets).
    del df_year, table, pos, neg, neg_sample, df_small
    gc.collect()

# Combine the sampled yearly data into one training DataFrame.
# This is the dataset I will use to fit models in Step 3.
train = pd.concat(train_parts, ignore_index=True)
print("\nTRAIN combined rows:", len(train))

# --------
# TEST = 2023 Jan–Jun (full, not sampled)
# --------
# Build a realistic test set: 2023 Jan–Jun, using ALL rows (no downsampling).
# This gives me an honest evaluation under the real class imbalance.
print("\nLoading TEST (2023 Jan–Jun)...")

start_test = arrow_ts(datetime(2023, 1, 1, tzinfo=timezone.utc))
end_test   = arrow_ts(datetime(2023, 7, 1, tzinfo=timezone.utc))

# Filter expression for the 2023H1 test window.
filt_test = (ds.field("landing_time") >= start_test) & (ds.field("landing_time") < end_test)

# Read the filtered test period into memory (still big, but only 6 months).
table_test = dataset.to_table(columns=keep_cols, filter=filt_test)
test = table_test.to_pandas()
test["GoA"] = test["GoA"].astype(int)

print("TEST rows:", len(test))
# Print go-around rates so I can see how sampling changes TRAIN vs the real-world TEST rate.
print("Go-around rate in TRAIN (sample):", train["GoA"].mean())
# Print go-around rates so I can see how sampling changes TRAIN vs the real-world TEST rate.
print("Go-around rate in TEST (real):", test["GoA"].mean())

# --------
# Add simple time features (hour/month/dayofweek)
# --------
# Feature engineering: apply the same transformations to both train and test for consistency.
for df in [train, test]:
    # Convert times to datetime
    # Convert landing_time to datetime so I can extract hour/month/day-of-week features.
    df["landing_time"] = pd.to_datetime(df["landing_time"], utc=True)
    # Convert C40 crossing time; errors='coerce' turns invalid values into NaT.
    df["C40_CROSS_TIME"] = pd.to_datetime(df["C40_CROSS_TIME"], utc=True, errors="coerce")

    # New numeric feature: minutes from 40nm crossing to landing
    # Minutes from 40NM crossing to landing.
    # I keep it initially, then explicitly test for leakage in Step 4 (since it uses landing_time).
    df["C40_to_landing_min"] = (df["landing_time"] - df["C40_CROSS_TIME"]).dt.total_seconds() / 60

    # Time-of-day features from landing time
    # Simple time features capturing time-of-day and seasonality patterns.
    df["hour"] = df["landing_time"].dt.hour
    df["month"] = df["landing_time"].dt.month
    df["dayofweek"] = df["landing_time"].dt.dayofweek

    # Drop datetime columns (models can’t use them)
    # Drop raw datetime columns (models cannot use datetime directly without encoding).
    df.drop(columns=["landing_time", "C40_CROSS_TIME"], inplace=True)
# -----------------------------
# Convert array-style categorical values into single values (leakage safety)
# -----------------------------
# In the raw parquet, some categorical fields (especially ILS) are stored as numpy arrays
# like: array(['09L'], dtype=object). Sometimes they can contain multiple values.
# If we keep them as arrays, the model can accidentally learn “attempt patterns”, which
# can inflate performance. To avoid that, I convert them into a single scalar value by
# keeping only the first element.

import numpy as np

def to_scalar(x):
    # If x is a numpy array (or list/tuple), keep only the first item
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0] if len(x) > 0 else None
    return x

# Apply the same cleaning to BOTH train and test so formats match
for df in [train, test]:
    for col in ["ILS", "AP_C_RWY", "market_segment", "AC_CLASS", "airport", "country"]:
        df[col] = df[col].apply(to_scalar)

# Quick check to confirm the cleaning worked (should be 0 after cleaning)
print("Array-like ILS AFTER cleaning (train):",
      train["ILS"].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).sum())


# Save the prepared datasets so training scripts do not need to re-scan the raw big file.
# Save smaller files so you don’t redo this
train.to_parquet("data/train_2019_2022_sample.parquet", index=False)
test.to_parquet("data/test_2023H1_full.parquet", index=False)

print("\nSaved:")
print("data/train_2019_2022_sample.parquet")
print("data/test_2023H1_full.parquet")
# I noticed from feature importance that some categorical fields (especially ILS)
# sometimes look like they contain multiple values (e.g., "['06L' '06L']").
# That usually happens when a cell is stored as a list/array instead of a single value.
#
# This can be a problem because lists/arrays might indirectly reflect multiple attempts,
# which could “leak” information related to a go-around.
#
# To make the model fair and consistent, I convert any list/tuple/array values into a
# single scalar value by keeping only the first element.

import numpy as np

def to_scalar(x):
    # If x is a list/tuple/numpy array, keep only the first item.
    # If it is empty, return None (so missing handling can deal with it later).
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0] if len(x) > 0 else None
    # If it's already a normal scalar (string/number), keep it as it is.
    return x

# Apply this cleaning to both training and test sets so they have the same format.
# I apply it to key categorical columns that should represent a single category.
for df in [train, test]:
    for col in ["ILS", "AP_C_RWY", "market_segment", "AC_CLASS", "airport", "country"]:
        df[col] = df[col].apply(to_scalar)
