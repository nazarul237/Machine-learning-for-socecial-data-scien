# STEP 6: Clean feature names for report readability
# --------------------------------------------------
# The raw feature names from Step 5 include prefixes like 'num__' and 'cat__'.
# This script makes a cleaner label (nice_feature), selects the top 20, and saves them to CSV.
# Load pandas for reading and writing CSV outputs.
import pandas as pd

# Load the full coefficient table created in Step 5.
df = pd.read_csv("results/logreg_feature_importance.csv")

# Make a nicer label
# Strip pipeline prefixes so the feature names are easy to interpret in the report.
df["nice_feature"] = (
    df["feature"]
    .str.replace("cat__", "", regex=False)
    .str.replace("num__", "", regex=False)
)

# Keep the top 20
# Select the 20 most influential features by absolute coefficient value.
top20 = df.sort_values("abs_coef", ascending=False).head(20)
print(top20[["nice_feature", "coef"]])

# Save the cleaned top-20 table for direct use in the report.
top20.to_csv("results/top20_nice_features.csv", index=False)
print("Saved: results/top20_nice_features.csv")
