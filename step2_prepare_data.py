import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from datetime import datetime, timezone
import gc

path = "data/osn23_landings_merged.parquet.gzip"

# Safe columns (no cheating with attempt count/times)
keep_cols = [
    "landing_time", "airport", "country",
    "ILS", "market_segment", "AC_CLASS",
    "AP_C_RWY",
    "C40_CROSS_TIME", "C40_BEARING", "C40_CROSS_LAT", "C40_CROSS_LON",
    "GoA"
]

NEG_RATIO = 20
RANDOM_STATE = 42

dataset = ds.dataset(path, format="parquet")

def arrow_ts(dt):
    # Make a timestamp that matches the dataset type (UTC timestamp)
    return pa.scalar(dt, type=pa.timestamp("ns", tz="UTC"))

train_parts = []

# --------
# TRAIN = 2019–2022 (sample negatives, keep all go-arounds)
# --------
for year in [2019, 2020, 2021, 2022]:
    print(f"\nLoading year {year}...")

    start = arrow_ts(datetime(year, 1, 1, tzinfo=timezone.utc))
    end   = arrow_ts(datetime(year + 1, 1, 1, tzinfo=timezone.utc))

    filt = (ds.field("landing_time") >= start) & (ds.field("landing_time") < end)

    table = dataset.to_table(columns=keep_cols, filter=filt)
    df_year = table.to_pandas()

    # Target 0/1
    df_year["GoA"] = df_year["GoA"].astype(int)

    pos = df_year[df_year["GoA"] == 1]
    neg = df_year[df_year["GoA"] == 0]

    n_neg = min(len(neg), len(pos) * NEG_RATIO)
    neg_sample = neg.sample(n=n_neg, random_state=RANDOM_STATE)

    df_small = pd.concat([pos, neg_sample]).sample(frac=1, random_state=RANDOM_STATE)

    print(f"Year {year}: total={len(df_year):,}  pos={len(pos):,}  neg_sample={len(neg_sample):,}  kept={len(df_small):,}")

    train_parts.append(df_small)

    # Free memory (important!)
    del df_year, table, pos, neg, neg_sample, df_small
    gc.collect()

train = pd.concat(train_parts, ignore_index=True)
print("\nTRAIN combined rows:", len(train))

# --------
# TEST = 2023 Jan–Jun (full, not sampled)
# --------
print("\nLoading TEST (2023 Jan–Jun)...")

start_test = arrow_ts(datetime(2023, 1, 1, tzinfo=timezone.utc))
end_test   = arrow_ts(datetime(2023, 7, 1, tzinfo=timezone.utc))

filt_test = (ds.field("landing_time") >= start_test) & (ds.field("landing_time") < end_test)

table_test = dataset.to_table(columns=keep_cols, filter=filt_test)
test = table_test.to_pandas()
test["GoA"] = test["GoA"].astype(int)

print("TEST rows:", len(test))
print("Go-around rate in TRAIN (sample):", train["GoA"].mean())
print("Go-around rate in TEST (real):", test["GoA"].mean())

# --------
# Add simple time features (hour/month/dayofweek)
# --------
for df in [train, test]:
    # Convert times to datetime
    df["landing_time"] = pd.to_datetime(df["landing_time"], utc=True)
    df["C40_CROSS_TIME"] = pd.to_datetime(df["C40_CROSS_TIME"], utc=True, errors="coerce")

    # New numeric feature: minutes from 40nm crossing to landing
    df["C40_to_landing_min"] = (df["landing_time"] - df["C40_CROSS_TIME"]).dt.total_seconds() / 60

    # Time-of-day features from landing time
    df["hour"] = df["landing_time"].dt.hour
    df["month"] = df["landing_time"].dt.month
    df["dayofweek"] = df["landing_time"].dt.dayofweek

    # Drop datetime columns (models can’t use them)
    df.drop(columns=["landing_time", "C40_CROSS_TIME"], inplace=True)
# Save smaller files so you don’t redo this
train.to_parquet("data/train_2019_2022_sample.parquet", index=False)
test.to_parquet("data/test_2023H1_full.parquet", index=False)

print("\nSaved:")
print("data/train_2019_2022_sample.parquet")
print("data/test_2023H1_full.parquet")