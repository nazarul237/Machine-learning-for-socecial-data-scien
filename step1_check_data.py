# STEP 1: Quick sanity check of the raw dataset
# ------------------------------------------------
# As an MSc student, I start by checking the raw file loads correctly.
# This script prints: (1) dataset size, (2) column names, (3) first few rows.
# It helps me confirm I am using the right file and that key fields exist.
import pandas as pd

# Load the raw dataset from the data/ folder (parquet is a binary table format).
# If this line fails, it usually means the file path is wrong or parquet support is missing.
df = pd.read_parquet("data/osn23_landings_merged.parquet.gzip")

# Print shape (rows, columns) so I understand dataset scale and whether it matches expectations.
print("df.shape =", df.shape)
# Print column names so I can decide which variables to use later for modelling.
print("Columns:", list(df.columns))
# Print a few example rows to visually check missing values and data types.
print(df.head(3))
