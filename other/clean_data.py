import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('C:/SIH/WQuality_River_Data_2023.csv')
    print("Successfully loaded the dataset.")
except FileNotFoundError:
    print("Error: WQuality_River_Data_2023.csv not found. Make sure the file is in the same directory.")
    exit()

# --- 1. Drop the Extraneous Column ---
# 'Column_21' is mostly empty and not useful.
if 'Column_21' in df.columns:
    df_cleaned = df.drop(columns=['Column_21'])
    print("Dropped 'Column_21'.")
else:
    df_cleaned = df.copy()

# --- 2. Remove Junk and Header Rows ---
# We can identify valid data rows by checking if 'Station_Code' is a number.
# Convert 'Station_Code' to a numeric type; anything that isn't a number will become 'NaN' (Not a Number).
df_cleaned['Station_Code'] = pd.to_numeric(df_cleaned['Station_Code'], errors='coerce')

# Now, drop all rows where 'Station_Code' is NaN.
rows_before = len(df_cleaned)
df_cleaned.dropna(subset=['Station_Code'], inplace=True)
rows_after = len(df_cleaned)
print(f"Removed {rows_before - rows_after} junk/header rows.")

# --- 3. Correct Data Types ---
# It's good practice to convert columns to their correct data types.
# 'Station_Code' should be an integer.
df_cleaned['Station_Code'] = df_cleaned['Station_Code'].astype(int)

# Identify all numeric columns (you can add or remove from this list as needed)
numeric_cols = [
    'Temperature_Min_C', 'Temperature_Max_C', 'Dissolved_Oxygen_Min_mgL',
    'Dissolved_Oxygen_Max_mgL', 'pH_Min', 'pH_Max', 'Conductivity_Min_umho_cm',
    'Conductivity_Max_umho_cm', 'BOD_Min_mgL', 'BOD_Max_mgL', 'Nitrate_N_Min_mgL',
    'Nitrate_N_Max_mgL', 'Fecal_Coliform_Min_MPN_100ml', 'Fecal_Coliform_Max_MPN_100ml',
    'Total_Coliform_Min_MPN_100ml', 'Total_Coliform_Max_MPN_100ml',
    'Fecal_Streptococci_Min_MPN_100ml', 'Fecal_Streptococci_Max_MPN_100ml'
]

# Convert these columns to numeric, coercing errors to NaN
for col in numeric_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

print("Corrected data types for numeric columns.")

# --- 4. Save the Cleaned Data ---
# Save the cleaned dataframe to a new CSV file to keep the original safe.
df_cleaned.to_csv('C:/SIH/WQuality_River_Data_2023_Cleaned.csv', index=False)
print("\nSuccessfully saved the cleaned data to 'WQuality_River_Data_2023_Cleaned.csv'")

# --- Display a summary of the cleaned data ---
print("\n--- Cleaned Data Summary ---")
print(f"Number of rows: {len(df_cleaned)}")
print(f"Number of columns: {len(df_cleaned.columns)}")
print("\nFirst 5 rows of the cleaned data:")
print(df_cleaned.head())
print("\nMissing values count in cleaned data:")
print(df_cleaned.isnull().sum())