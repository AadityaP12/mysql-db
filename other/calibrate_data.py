import pandas as pd
import numpy as np

# Load current v3 dataset (the one you already generated)
file_path = "C:/SIH/northeast_india_waterborne_disease_dataset_v3.csv"
df = pd.read_csv(file_path)

# -----------------------------
# 1. Refine rainfall realism (fixed)
# -----------------------------
def adjust_rainfall_v3(val, state):
    if state == "Meghalaya":
        # Keep rare natural extremes if originally high
        if val > 300 and np.random.rand() < 0.02:
            return min(val, 900)   # cap rare Meghalaya outliers at 900
        return min(val, 300)       # most Meghalaya days capped at 300
    else:
        return min(val, 300)       # all other states capped at 300

if "State" in df.columns:
    df["Rainfall_mm"] = df.apply(lambda row: adjust_rainfall_v3(row["Rainfall_mm"], row["State"]), axis=1)
else:
    df["Rainfall_mm"] = df["Rainfall_mm"].apply(lambda x: min(x, 300))

# -----------------------------
# 2. (Optional) Keep treatment, outcome, recall bias as is
# Since v3 already fixed them, we won’t re-randomize again
# -----------------------------

# -----------------------------
# 3. Save Updated v3 (capped rainfall)
# -----------------------------
updated_path = "C:/SIH/northeast_india_waterborne_disease_dataset_v3_capped.csv"
df.to_csv(updated_path, index=False)

print(f"Dataset 3 (capped rainfall) saved to: {updated_path}")
