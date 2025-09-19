import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("C:/SIH/northeast_india_waterborne_disease_dataset_v3_capped.csv")

# Function to detect discrete vs continuous
def is_discrete(series, threshold=20):
    """Return True if numeric column seems discrete (low number of unique values)."""
    return series.nunique() <= threshold

for col in df.columns:
    if df[col].isnull().sum() > 0:  # only handle columns with missing values
        if pd.api.types.is_numeric_dtype(df[col]):
            if is_discrete(df[col]):
                # Fill discrete numeric with median
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Fill continuous numeric with mean
                df[col].fillna(df[col].mean(), inplace=True)
        else:
            # Fill categorical with mode
            mode_val = df[col].mode().iloc[0]
            df[col].fillna(mode_val, inplace=True)

# Save cleaned dataset
df.to_csv("C:/SIH/northeast_india_waterborne_disease_dataset_v3_cleaned.csv", index=False)

print("✅ Missing values handled and dataset saved as 'northeast_india_waterborne_disease_dataset_v3_cleaned.csv'")
