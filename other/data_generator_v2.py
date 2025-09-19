import pandas as pd
import numpy as np
import random
import re
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load the simulated dataset
df = pd.read_csv("C:/SIH/northeast_india_waterborne_disease_dataset.csv")

# Load the OCR health infrastructure report
ocr = open("ocr_output_2.txt", encoding="utf-8").read()

# Extract village-to-coordinate mappings from OCR (example pattern: "Village_5 (Lat:27.XXX, Lon:95.XXX)")
village_coords = {}
for match in re.finditer(r"Village_(\d+)[\s\S]{0,50}?Lat[:]?(\d+\.\d+)[\s,]+Lon[:]?(\d+\.\d+)", ocr):
    vid, lat, lon = match.groups()
    village_coords[f"Village_{vid}"] = (float(lat), float(lon))

# Helper to assign coordinates
def assign_coords(village):
    return village_coords.get(village, (np.nan, np.nan))

# Enhance dataset with realistic addresses and geo-coordinates
def enhance_dataset(df):
    df = df.copy()
    # Assign precise coords
    coords = df["Village"].apply(assign_coords).tolist()
    df["Latitude"], df["Longitude"] = zip(*coords)

    # Generate realistic street addresses within each village
    df["House_Number"] = df.groupby(["State","District","Village"]).cumcount() + 1
    df["Street_Name"] = df["Village"].str.replace("_", " ") + " Main Road"
    df["Address"] = df["House_Number"].astype(str) + ", " + df["Street_Name"] + ", " + df["District"] + ", " + df["State"]

    # Add sanitation facility and source contamination flag from OCR context
    # Assume any village with Ecoli_Count_cfu > 200 is 'Source Contaminated'
    df["Source_Contaminated"] = df["Ecoli_Count_cfu"] > 200

    # Add household sanitation score from OCR: parse sanitation coverage stats per district
    sanitation_scores = {}
    for m in re.finditer(rf"{re.escape(',')} {re.escape('sanitation')}\s+coverage\s+in\s+([\w\s]+):\s+(\d+)%", ocr):
        district, pct = m.groups()
        sanitation_scores[district.strip()] = int(pct)/100
    df["Sanitation_Coverage"] = df["District"].map(sanitation_scores).fillna(0.75)

    # Compute individual infection risk score
    df["Infection_Risk"] = (
        df["Ecoli_Count_cfu"] / 500 * 
        df["Sanitation_Coverage"] * 
        df["Source_Contaminated"].astype(int)
    ).clip(0,1).round(2)

    # Predict outbreak before symptoms: flag if risk>0.6 and Outcome == 'Recovered'
    df["Likely_Pre_Symptom_Outbreak"] = (df["Infection_Risk"] > 0.6) & (df["Outcome"] == "Recovered")

    return df

enhanced = enhance_dataset(df)
enhanced.to_csv("C:/SIH/enhanced_waterborne_health_data.csv", index=False)
print("Enhanced dataset created: enhanced_waterborne_health_data.csv")
