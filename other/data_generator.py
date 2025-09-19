import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Northeast India states and realistic data
NE_STATES_DATA = {
    'Assam': {
        'districts': ['Kamrup', 'Dibrugarh', 'Guwahati', 'Jorhat', 'Silchar', 'Tezpur', 'Nagaon', 'Golaghat', 'Sivasagar', 'Barpeta'],
        'villages_per_district': 25,
        'terrain': 'riverine_plains',
        'avg_rainfall': 180,
        'water_sources': ['River', 'Pond', 'Tube Well', 'Well', 'Stream']
    },
    'Arunachal Pradesh': {
        'districts': ['Itanagar', 'Tawang', 'Bomdila', 'Ziro', 'Pasighat', 'Tezu', 'Changlang', 'Khonsa', 'Seppa', 'Roing'],
        'villages_per_district': 20,
        'terrain': 'mountainous',
        'avg_rainfall': 220,
        'water_sources': ['Spring', 'River', 'Tube Well', 'Well']
    },
    'Manipur': {
        'districts': ['Imphal East', 'Imphal West', 'Churachandpur', 'Thoubal', 'Bishnupur', 'Chandel', 'Senapati', 'Tamenglong', 'Ukhrul', 'Jiribam'],
        'villages_per_district': 22,
        'terrain': 'valley_hills',
        'avg_rainfall': 160,
        'water_sources': ['Pond', 'Tube Well', 'Spring', 'River', 'Well']
    },
    'Meghalaya': {
        'districts': ['East Khasi Hills', 'West Khasi Hills', 'Ri-Bhoi', 'East Garo Hills', 'West Garo Hills', 'South Garo Hills', 'Jaintia Hills'],
        'villages_per_district': 24,
        'terrain': 'plateau',
        'avg_rainfall': 280,
        'water_sources': ['Spring', 'River', 'Well', 'Tube Well']
    },
    'Mizoram': {
        'districts': ['Aizawl', 'Lunglei', 'Saiha', 'Champhai', 'Kolasib', 'Serchhip', 'Lawngtlai', 'Mamit'],
        'villages_per_district': 18,
        'terrain': 'hills',
        'avg_rainfall': 200,
        'water_sources': ['Spring', 'River', 'Well', 'Tube Well']
    },
    'Nagaland': {
        'districts': ['Kohima', 'Dimapur', 'Mokokchung', 'Tuensang', 'Wokha', 'Zunheboto', 'Phek', 'Mon', 'Longleng', 'Kiphire'],
        'villages_per_district': 20,
        'terrain': 'hills',
        'avg_rainfall': 190,
        'water_sources': ['Spring', 'River', 'Well', 'Tube Well']
    },
    'Sikkim': {
        'districts': ['East Sikkim', 'West Sikkim', 'North Sikkim', 'South Sikkim'],
        'villages_per_district': 15,
        'terrain': 'mountainous',
        'avg_rainfall': 250,
        'water_sources': ['Spring', 'River', 'Well', 'Tube Well']
    },
    'Tripura': {
        'districts': ['West Tripura', 'South Tripura', 'Dhalai', 'North Tripura', 'Khowai', 'Gomati', 'Unakoti', 'Sepahijala'],
        'villages_per_district': 25,
        'terrain': 'hills_plains',
        'avg_rainfall': 170,
        'water_sources': ['River', 'Pond', 'Tube Well', 'Well']
    }
}

# Water-borne diseases with realistic patterns
DISEASES = {
    'Diarrhea': {
        'incubation_days': [1, 2, 3],
        'severity': 'mild_moderate',
        'season_preference': ['monsoon', 'post_monsoon'],
        'contamination_threshold': 0.4
    },
    'Cholera': {
        'incubation_days': [1, 2, 3, 4, 5],
        'severity': 'severe',
        'season_preference': ['monsoon', 'post_monsoon'],
        'contamination_threshold': 0.6
    },
    'Typhoid': {
        'incubation_days': [7, 10, 14, 21],
        'severity': 'moderate_severe',
        'season_preference': ['pre_monsoon', 'monsoon'],
        'contamination_threshold': 0.5
    },
    'Hepatitis A': {
        'incubation_days': [15, 21, 28, 35],
        'severity': 'moderate_severe',
        'season_preference': ['pre_monsoon', 'monsoon', 'post_monsoon'],
        'contamination_threshold': 0.5
    }
}

# Treatment options and outcomes
TREATMENTS = ['None', 'ORS', 'Antibiotics', 'Hospital Referral']
OUTCOMES = ['Recovered', 'Under Treatment', 'Referred', 'Death']
REPORTERS = ['ASHA', 'Clinic', 'Volunteer']

def get_season(date):
    """Determine season based on date"""
    month = date.month
    if month in [3, 4, 5]:
        return 'pre_monsoon'
    elif month in [6, 7, 8, 9]:
        return 'monsoon'
    elif month in [10, 11]:
        return 'post_monsoon'
    else:
        return 'winter'

def calculate_contamination_risk(water_source, season, rainfall, terrain):
    """Calculate water contamination risk based on multiple factors"""
    base_risk = 0.3
    
    # Water source risk
    source_risk = {
        'River': 0.7, 'Pond': 0.8, 'Well': 0.4, 
        'Tube Well': 0.2, 'Spring': 0.3
    }
    
    base_risk = source_risk.get(water_source, 0.5)
    
    # Seasonal adjustments
    if season == 'monsoon':
        base_risk *= 1.4
    elif season == 'post_monsoon':
        base_risk *= 1.2
    
    # Terrain adjustments
    if terrain == 'riverine_plains':
        base_risk *= 1.2
    elif terrain == 'mountainous':
        base_risk *= 0.8
    
    # Rainfall effect
    if rainfall > 200:
        base_risk *= 1.3
    elif rainfall > 150:
        base_risk *= 1.1
    
    return min(base_risk, 1.0)

def generate_water_quality_parameters(contamination_risk, water_source):
    """Generate realistic water quality parameters"""
    
    # pH - normal range 6.5-8.5
    if contamination_risk > 0.6:
        ph = np.random.choice([
            np.random.normal(5.8, 0.4),  # Acidic contamination
            np.random.normal(8.7, 0.3)   # Basic contamination
        ])
    else:
        ph = np.random.normal(7.2, 0.5)
    
    ph = np.clip(ph, 5.5, 9.0)
    
    # Turbidity - higher in contaminated sources
    if contamination_risk > 0.7:
        turbidity = np.random.exponential(15)
    elif contamination_risk > 0.4:
        turbidity = np.random.exponential(8)
    else:
        turbidity = np.random.exponential(3)
    
    turbidity = np.clip(turbidity, 0.5, 20)
    
    # E.coli count - directly related to contamination risk
    if contamination_risk > 0.7:
        ecoli = np.random.exponential(300)
    elif contamination_risk > 0.4:
        ecoli = np.random.exponential(150)
    else:
        ecoli = np.random.exponential(50)
    
    ecoli = int(np.clip(ecoli, 0, 500))
    
    # Adjust for water source type
    if water_source in ['Spring', 'Tube Well']:
        ecoli = int(ecoli * 0.5)
        turbidity = turbidity * 0.6
    
    return round(ph, 2), round(turbidity, 2), ecoli

def should_get_disease(contamination_risk, age, season):
    """Determine if a person should get a disease based on risk factors"""
    
    # Age-based vulnerability
    if age < 5 or age > 65:
        age_factor = 1.4
    elif age < 15:
        age_factor = 1.1
    else:
        age_factor = 1.0
    
    # Seasonal factor
    seasonal_factor = 1.2 if season in ['monsoon', 'post_monsoon'] else 1.0
    
    # Combined probability
    disease_prob = contamination_risk * age_factor * seasonal_factor * 0.4  # Base 40% attack rate
    
    return random.random() < disease_prob

def select_disease(season, contamination_risk):
    """Select appropriate disease based on conditions"""
    suitable_diseases = []
    
    for disease, info in DISEASES.items():
        if (season in info['season_preference'] and 
            contamination_risk >= info['contamination_threshold']):
            suitable_diseases.append(disease)
    
    if not suitable_diseases:
        suitable_diseases = ['Diarrhea']  # Fallback
    
    return random.choice(suitable_diseases)

def generate_realistic_dataset(num_records=7000):
    """Generate the exact dataset format requested"""
    
    print("Generating realistic water-borne disease dataset...")
    all_records = []
    
    # Date range: 2023 full year
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    records_per_state = num_records // len(NE_STATES_DATA)
    
    for state, state_data in NE_STATES_DATA.items():
        print(f"Generating data for {state}...")
        
        state_records = []
        districts = state_data['districts']
        water_sources = state_data['water_sources']
        
        while len(state_records) < records_per_state:
            # Random date
            random_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            
            # Random location
            district = random.choice(districts)
            village_id = random.randint(1, state_data['villages_per_district'])
            village_name = f"Village_{village_id}"
            
            # Random water source
            water_source = random.choice(water_sources)
            
            # Calculate season and contamination
            season = get_season(random_date)
            contamination_risk = calculate_contamination_risk(
                water_source, season, state_data['avg_rainfall'], state_data['terrain']
            )
            
            # Generate water quality
            ph, turbidity, ecoli = generate_water_quality_parameters(contamination_risk, water_source)
            
            # Person characteristics
            age = max(1, int(np.random.exponential(25)))
            age = min(age, 80)
            gender = random.choice(['Male', 'Female'])
            
            # Determine if this person gets a disease
            gets_disease = should_get_disease(contamination_risk, age, season)
            
            if gets_disease:
                # Select disease
                disease = select_disease(season, contamination_risk)
                
                # Environmental factors
                rainfall = max(0, np.random.normal(state_data['avg_rainfall'], 30))
                
                if season == 'winter':
                    temperature = np.random.normal(15, 5)
                elif season == 'pre_monsoon':
                    temperature = np.random.normal(28, 6)
                elif season == 'monsoon':
                    temperature = np.random.normal(26, 4)
                else:  # post_monsoon
                    temperature = np.random.normal(24, 5)
                
                temperature = np.clip(temperature, 10, 35)
                
                # Treatment and outcome
                treatment_probs = [0.1, 0.4, 0.3, 0.2]  # None, ORS, Antibiotics, Hospital
                treatment = np.random.choice(TREATMENTS, p=treatment_probs)
                
                # Outcome based on age, treatment, and disease severity
                if age < 5 or age > 70:
                    outcome_probs = [0.6, 0.25, 0.1, 0.05]  # Higher risk for vulnerable
                else:
                    outcome_probs = [0.75, 0.2, 0.04, 0.01]
                
                if treatment == 'Hospital Referral':
                    outcome_probs = [0.8, 0.15, 0.04, 0.01]  # Better with hospital care
                elif treatment == 'None':
                    outcome_probs = [0.5, 0.3, 0.15, 0.05]  # Worse without treatment
                
                outcome = np.random.choice(OUTCOMES, p=outcome_probs)
                
                # Reporter
                reporter_probs = [0.5, 0.3, 0.2]  # ASHA, Clinic, Volunteer
                reporter = np.random.choice(REPORTERS, p=reporter_probs)
                
                # Create record with exact columns requested
                record = {
                    'Record_ID': len(all_records) + 1,
                    'Date': random_date.strftime('%Y-%m-%d'),
                    'State': state,
                    'District': district,
                    'Village': village_name,
                    'Patient_Age': age,
                    'Gender': gender,
                    'Disease_Symptom': disease,
                    'Water_Source': water_source,
                    'pH': ph,
                    'Turbidity_NTU': turbidity,
                    'Ecoli_Count_cfu': ecoli,
                    'Rainfall_mm': round(rainfall, 1),
                    'Temperature_C': round(temperature, 1),
                    'Reported_By': reporter,
                    'Treatment_Given': treatment,
                    'Outcome': outcome
                }
                
                state_records.append(record)
        
        all_records.extend(state_records[:records_per_state])
        print(f"Generated {len(state_records[:records_per_state])} records for {state}")
    
    print(f"Total records generated: {len(all_records)}")
    return all_records

def add_outbreak_clusters(records, num_clusters=12):
    """Add realistic disease outbreak clusters"""
    print("Adding outbreak clusters...")
    
    cluster_records = []
    base_records = random.sample(records, num_clusters)
    
    for cluster_id, base_record in enumerate(base_records):
        cluster_size = random.randint(15, 40)
        base_date = datetime.strptime(base_record['Date'], '%Y-%m-%d')
        
        for i in range(cluster_size):
            # Similar date (within 3 weeks)
            date_offset = random.randint(-21, 21)
            cluster_date = base_date + timedelta(days=date_offset)
            
            if cluster_date.year != 2023:
                continue
            
            # Similar location and water source
            age = max(1, int(np.random.exponential(25)))
            age = min(age, 80)
            gender = random.choice(['Male', 'Female'])
            
            # Higher contamination for outbreak
            contamination_risk = min(0.9, random.uniform(0.7, 0.9))
            
            # More severe water parameters
            ph = np.random.choice([
                np.random.normal(5.5, 0.3),
                np.random.normal(8.9, 0.3)
            ])
            turbidity = np.random.exponential(18)
            ecoli = int(np.random.exponential(350))
            
            ph = np.clip(ph, 5.5, 9.0)
            turbidity = np.clip(turbidity, 5, 20)
            ecoli = np.clip(ecoli, 100, 500)
            
            # High probability of severe disease
            if random.random() < 0.85:  # 85% attack rate in outbreak
                disease = np.random.choice(['Cholera', 'Typhoid', 'Hepatitis A'], p=[0.4, 0.3, 0.3])
                
                # Treatment and outcomes
                treatment = np.random.choice(TREATMENTS, p=[0.05, 0.25, 0.4, 0.3])
                
                if age < 5 or age > 65:
                    outcome_probs = [0.5, 0.35, 0.1, 0.05]
                else:
                    outcome_probs = [0.7, 0.25, 0.04, 0.01]
                
                outcome = np.random.choice(OUTCOMES, p=outcome_probs)
                reporter = np.random.choice(REPORTERS, p=[0.6, 0.3, 0.1])
                
                # Environmental factors similar to base
                rainfall = max(0, np.random.normal(float(base_record['Rainfall_mm']), 20))
                temperature = np.random.normal(float(base_record['Temperature_C']), 3)
                temperature = np.clip(temperature, 10, 35)
                
                cluster_record = {
                    'Record_ID': len(records) + len(cluster_records) + 1,
                    'Date': cluster_date.strftime('%Y-%m-%d'),
                    'State': base_record['State'],
                    'District': base_record['District'],
                    'Village': base_record['Village'],
                    'Patient_Age': age,
                    'Gender': gender,
                    'Disease_Symptom': disease,
                    'Water_Source': base_record['Water_Source'],
                    'pH': round(ph, 2),
                    'Turbidity_NTU': round(turbidity, 2),
                    'Ecoli_Count_cfu': int(ecoli),
                    'Rainfall_mm': round(rainfall, 1),
                    'Temperature_C': round(temperature, 1),
                    'Reported_By': reporter,
                    'Treatment_Given': treatment,
                    'Outcome': outcome
                }
                
                cluster_records.append(cluster_record)
    
    print(f"Added {len(cluster_records)} outbreak records")
    return records + cluster_records

def main():
    """Main function to generate and save dataset"""
    
    # Generate main dataset
    print("Starting dataset generation...")
    dataset_records = generate_realistic_dataset(num_records=5000)
    
    # Add outbreak clusters
    dataset_records = add_outbreak_clusters(dataset_records, num_clusters=15)
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset_records)
    
    # Sort by date and reset Record_ID
    df = df.sort_values('Date').reset_index(drop=True)
    df['Record_ID'] = range(1, len(df) + 1)
    
    # Ensure we have the exact columns requested
    required_columns = [
        'Record_ID', 'Date', 'State', 'District', 'Village', 'Patient_Age', 
        'Gender', 'Disease_Symptom', 'Water_Source', 'pH', 'Turbidity_NTU', 
        'Ecoli_Count_cfu', 'Rainfall_mm', 'Temperature_C', 'Reported_By', 
        'Treatment_Given', 'Outcome'
    ]
    
    df = df[required_columns]
    
    # Save to the specified path
    output_path = r"C:\SIH\northeast_india_waterborne_disease_dataset.csv"
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save dataset
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved successfully to: {output_path}")
        
        # Display statistics
        print(f"\n=== DATASET STATISTICS ===")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"States: {df['State'].nunique()}")
        print(f"Districts: {df['District'].nunique()}")
        print(f"Villages: {df['Village'].nunique()}")
        
        print(f"\nDisease distribution:")
        print(df['Disease_Symptom'].value_counts())
        
        print(f"\nWater source distribution:")
        print(df['Water_Source'].value_counts())
        
        print(f"\nOutcome distribution:")
        print(df['Outcome'].value_counts())
        
        # Display sample records
        print(f"\n=== SAMPLE RECORDS ===")
        print(df.head(5).to_string(index=False))
        
        print(f"\nDataset generation completed successfully!")
        print(f"Ready for ML model training and water-borne disease prediction.")
        
    except Exception as e:
        print(f"Error saving file: {e}")
        print("Saving to current directory instead...")
        df.to_csv("waterborne_disease_dataset.csv", index=False)
        print("Dataset saved as: waterborne_disease_dataset.csv")

if __name__ == "__main__":
    main()