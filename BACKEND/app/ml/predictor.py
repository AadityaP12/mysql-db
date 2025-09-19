import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import logging
from pathlib import Path

from app.config import settings
from app.core.monitoring import structured_logger
from app.core.utils import convert_to_serializable

logger = structured_logger

from app.config import settings
import os

print("🔍 MODEL_PATH:", settings.MODEL_PATH)
print("📂 Available files:", os.listdir(settings.MODEL_PATH))


class WaterQualityModel:
    """Water Quality Risk Prediction Model"""
    
    def __init__(self):
        self.model = None
        self.model_path = Path(settings.MODEL_PATH) / "rf_water_model_v2.pkl"
        self.fallback_models = {
            "xgb": Path(settings.MODEL_PATH) / "xgb_water_model_v2.pkl"
        }
        self.is_loaded = False
        # CORRECTED feature columns based on model inspection
        self.feature_columns = [
            'source_type',
            'rainfall_24h_mm', 
            'temperature_C',
            'dissolved_oxygen_mgL',
            'chlorine_mgL',
            'month',
            'fecal_coliform_MPN',
            'season',
            'pH',
            'turbidity_NTU',
            'persons_with_symptoms'
        ]
        self.required_features = ['pH', 'turbidity_NTU', 'source_type']
        
        # CORRECTED categorical values from model inspection
        self.valid_source_types = [
            'community_tank', 'deep_borehole', 'open_catchment', 
            'piped_protected', 'pond', 'reservoir', 'river',
            'rooftop_rainwater', 'shallow_well', 'spring'
        ]
        self.valid_seasons = ['Monsoon', 'Summer', 'Winter']  # Note capitalization
        self.valid_months = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
    def load_model(self) -> bool:
        """Load the trained water quality model"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                logger.info("Water quality model loaded successfully", model_path=str(self.model_path))
                return True
            else:
                # Try loading fallback model
                for model_name, path in self.fallback_models.items():
                    if path.exists():
                        self.model = joblib.load(path)
                        self.is_loaded = True
                        logger.info(f"Fallback water quality model ({model_name}) loaded", model_path=str(path))
                        return True
                
                logger.error("No water quality model found", 
                           primary_path=str(self.model_path),
                           fallback_paths=[str(p) for p in self.fallback_models.values()])
                return False
                
        except Exception as e:
            logger.error("Failed to load water quality model", error=str(e))
            return False
    
    def _map_water_source(self, api_water_source: str) -> str:
        """Map API water source values to model categories"""
        mapping = {
            'piped_water': 'piped_protected',
            'piped': 'piped_protected', 
            'tap_water': 'piped_protected',
            'well_water': 'shallow_well',
            'well': 'shallow_well',
            'groundwater': 'deep_borehole',
            'surface_water': 'river',
            'surface': 'river',  # FIXED: added this mapping
            'river_water': 'river',
            'spring_water': 'spring',
            'rainwater': 'rooftop_rainwater',
            'tank_water': 'community_tank',
            'pond_water': 'pond',
            'reservoir_water': 'reservoir',
            'catchment': 'open_catchment',
            'borehole': 'deep_borehole'
        }
        
        # Default mapping
        mapped = mapping.get(api_water_source.lower(), 'piped_protected')
        
        # Ensure it's a valid category
        if mapped not in self.valid_source_types:
            logger.warning(f"Invalid source type '{mapped}', using default 'piped_protected'")
            return 'piped_protected'  # Safe default
        
        return mapped
    
    def _map_season(self, api_season: str) -> str:
        """Map API season values to model categories"""
        mapping = {
            'summer': 'Summer',  # FIXED: proper capitalization
            'monsoon': 'Monsoon', 
            'rainy': 'Monsoon',
            'winter': 'Winter',
            'cold': 'Winter'
        }
        
        mapped = mapping.get(api_season.lower(), 'Summer')
        
        # Ensure it's valid
        if mapped not in self.valid_seasons:
            logger.warning(f"Invalid season '{mapped}', using default 'Summer'")
            return 'Summer'  # Safe default
        
        return mapped
    
    def _get_current_season_and_month(self) -> tuple:
        """Get current season and month based on date"""
        current_month = datetime.now().month
        
        # Indian seasons based on months
        if current_month in [12, 1, 2]:
            season = 'Winter'
        elif current_month in [3, 4, 5]:
            season = 'Summer' 
        else:  # 6-11
            season = 'Monsoon'
        
        # Ensure month is in valid range (model expects 2-12, no month 1)
        if current_month not in self.valid_months:
            if current_month == 1:
                # --- IMPROVEMENT ---
                # Added a comment to clarify why January (1) is mapped to February (2).
                # This is intentional because the training data for the model
                # only includes months 2 through 12.
                logger.info("Mapping month 1 (January) to 2 (February) to match model's trained feature set.")
                current_month = 2  # Map January to February
            else:
                current_month = 6  # Default to June
            
        return season, current_month
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for the model"""
        try:
            # Get current season and month if not provided
            current_season, current_month = self._get_current_season_and_month()
            
            # CORRECTED feature mapping with proper validation
            processed_data = {
                'source_type': self._map_water_source(input_data.get('water_source', 'piped_water')),
                'rainfall_24h_mm': max(0.0, float(input_data.get('rainfall', 0.0))),
                'temperature_C': max(0.0, float(input_data.get('temperature', 25.0))),
                'dissolved_oxygen_mgL': self._estimate_dissolved_oxygen(input_data),
                'chlorine_mgL': max(0.0, float(input_data.get('chloramines', 0.5))),
                'month': input_data.get('month', current_month),
                'fecal_coliform_MPN': self._estimate_fecal_coliform(input_data),
                'season': self._map_season(input_data.get('season', current_season.lower())),
                'pH': max(0.0, min(14.0, float(input_data.get('ph_level', 7.0)))),
                'turbidity_NTU': max(0.0, float(input_data.get('turbidity', 1.8))),
                'persons_with_symptoms': max(0.0, float(input_data.get('population_density', 250) * 0.001)),
            }
            
            # Convert to DataFrame
            df = pd.DataFrame([processed_data])
            
            # Ensure all columns are present and in correct order
            for col in self.feature_columns:
                if col not in df.columns:
                    # Set safe defaults for any missing columns
                    if col == 'source_type':
                        df[col] = 'piped_protected'
                    elif col == 'season':
                        df[col] = 'Summer'
                    elif col == 'month':
                        df[col] = 6
                    elif col == 'pH':
                        df[col] = 7.0
                    elif col == 'turbidity_NTU':
                        df[col] = 1.8
                    elif col == 'temperature_C':
                        df[col] = 25.0
                    elif col == 'dissolved_oxygen_mgL':
                        df[col] = 8.0
                    elif col == 'chlorine_mgL':
                        df[col] = 0.5
                    elif col == 'fecal_coliform_MPN':
                        df[col] = 2.0
                    elif col == 'rainfall_24h_mm':
                        df[col] = 0.0
                    elif col == 'persons_with_symptoms':
                        df[col] = 1.0
                    else:
                        df[col] = 0.0
            
            # Ensure correct column order
            df = df[self.feature_columns]
            
            # CRITICAL: Final validation of categorical values
            source_type = df['source_type'].iloc[0]
            if source_type not in self.valid_source_types:
                logger.warning(f"Invalid source_type '{source_type}', replacing with 'piped_protected'")
                df['source_type'].iloc[0] = 'piped_protected'
            
            season = df['season'].iloc[0]
            if season not in self.valid_seasons:
                logger.warning(f"Invalid season '{season}', replacing with 'Summer'")
                df['season'].iloc[0] = 'Summer'
                
            month = df['month'].iloc[0]
            if month not in self.valid_months:
                logger.warning(f"Invalid month '{month}', replacing with 6")
                df['month'].iloc[0] = 6
            
            # Final data type validation
            # Ensure numeric columns are float
            numeric_cols = ['rainfall_24h_mm', 'temperature_C', 'dissolved_oxygen_mgL', 
                           'chlorine_mgL', 'fecal_coliform_MPN', 'pH', 'turbidity_NTU', 
                           'persons_with_symptoms']
            for col in numeric_cols:
                df[col] = df[col].astype(float)
            
            # Ensure month is int
            df['month'] = df['month'].astype(int)

            # --- IMPROVEMENT ---
            # Added logging for key numeric feature ranges to help debug edge cases.
            logger.info("Model input validated", 
                       source_type=df['source_type'].iloc[0],
                       season=df['season'].iloc[0],
                       month=df['month'].iloc[0],
                       ph_value=df['pH'].iloc[0],
                       turbidity_value=df['turbidity_NTU'].iloc[0],
                       temp_value=df['temperature_C'].iloc[0])
            
            return df
            
        except Exception as e:
            logger.error("Error preprocessing water quality input", error=str(e))
            raise
    
    def _estimate_dissolved_oxygen(self, input_data: Dict[str, Any]) -> float:
        """Estimate dissolved oxygen based on available parameters"""
        temp = float(input_data.get('temperature', 25.0))
        ph = float(input_data.get('ph_level', 7.0))
        
        # Cooler water holds more oxygen, pH affects solubility
        base_do = 9.0  # mg/L at 25°C
        temp_factor = max(0.5, 1.0 - (temp - 25) * 0.02)
        ph_factor = 1.0 - abs(ph - 7.0) * 0.05
        
        result = max(3.0, min(12.0, base_do * temp_factor * ph_factor))
        return float(result)
    
    def _estimate_fecal_coliform(self, input_data: Dict[str, Any]) -> float:
        """Estimate fecal coliform based on sanitation and water source"""
        sanitation = input_data.get('sanitation', 'improved')
        water_source = input_data.get('water_source', 'piped_water')
        turbidity = float(input_data.get('turbidity', 1.0))
        
        base_coliform = 2.0  # Log MPN/100ml baseline
        
        # Sanitation impact
        if sanitation == 'unimproved':
            base_coliform += 1.5
        elif sanitation == 'open_defecation':
            base_coliform += 2.5
        
        # Water source impact
        if 'surface' in water_source.lower() or 'river' in water_source.lower():
            base_coliform += 1.0
        elif 'well' in water_source.lower():
            base_coliform += 0.5
        
        # Turbidity correlation
        turbidity_factor = min(2.0, turbidity * 0.3)
        
        result = max(0.0, min(6.0, base_coliform + turbidity_factor))
        return float(result)
    
    def predict_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict water quality risk"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Water quality model not available")
        
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Debug: Log the actual input to the model
            logger.info("Model input prepared", 
                       input_shape=X.shape,
                       source_type=X['source_type'].iloc[0],
                       season=X['season'].iloc[0],
                       month=X['month'].iloc[0])
            
            # Make prediction (returns risk percentage 0-100)
            risk_percent = float(self.model.predict(X)[0])
            risk_score = max(0.0, min(1.0, risk_percent / 100.0))  # Normalize to 0-1
            
            # Determine risk level
            if risk_score < 0.3:
                risk_level = "LOW"
                risk_color = "GREEN"
            elif risk_score < 0.7:
                risk_level = "MEDIUM"
                risk_color = "YELLOW"
            else:
                risk_level = "HIGH"
                risk_color = "RED"
            
            # Generate risk factors
            risk_factors = self._identify_risk_factors(input_data, X)
            
            return {
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "risk_percent": float(risk_percent),
                "confidence": 0.85,
                "primary_risk_factors": risk_factors,
                "model_version": "rf_water_v2.0_corrected",
                "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_input_features": {
                    "source_type": X['source_type'].iloc[0],
                    "season": X['season'].iloc[0], 
                    "month": int(X['month'].iloc[0]),
                    "pH": float(X['pH'].iloc[0]),
                    "turbidity_NTU": float(X['turbidity_NTU'].iloc[0])
                }
            }
            
        except Exception as e:
            logger.error("Water quality prediction failed", error=str(e))
            raise
    
    def _identify_risk_factors(self, original_data: Dict[str, Any], processed_data: pd.DataFrame) -> List[str]:
        """Identify main risk factors based on input values"""
        factors = []
        
        ph = processed_data['pH'].iloc[0]
        turbidity = processed_data['turbidity_NTU'].iloc[0] 
        do_level = processed_data['dissolved_oxygen_mgL'].iloc[0]
        coliform = processed_data['fecal_coliform_MPN'].iloc[0]
        source_type = processed_data['source_type'].iloc[0]
        
        if ph < 6.5 or ph > 8.5:
            factors.append("pH levels outside safe range")
        
        if turbidity > 4.0:
            factors.append("High turbidity levels")
        elif turbidity > 2.0:
            factors.append("Elevated turbidity")
        
        if do_level < 5.0:
            factors.append("Low dissolved oxygen levels")
        
        if coliform > 3.0:
            factors.append("High microbial contamination risk")
        
        if source_type in ['pond', 'river', 'open_catchment']:
            factors.append("Higher risk water source")
        
        if original_data.get('sanitation') in ['unimproved', 'open_defecation']:
            factors.append("Poor sanitation infrastructure")
        
        if not factors:
            factors.append("General water quality parameters")
        
        return factors


class HealthRiskModel:
    def __init__(self):
        self.model = None
        self.model_path = Path(settings.MODEL_PATH) / "rf_health_model.pkl"
        self.fallback_models = {
            "lgb": Path(settings.MODEL_PATH) / "lgb_health_model.pkl",
            "xgb": Path(settings.MODEL_PATH) / "xgb_health_model.pkl"
        }
        self.is_loaded = False
        
        # The raw feature columns that the pipeline expects
        # Based on your training script, these are the columns before preprocessing
        self.raw_feature_columns = [
            'age', 'sex', 'sanitation', 'water_source',
            'diarrhea', 'vomiting', 'fever', 'fatigue', 
            'jaundice', 'headache', 'loss_appetite', 'muscle_aches'
        ]

    def load_model(self) -> bool:
        """Load the trained health risk model"""
        try:
            # Try to load the primary model first
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                logger.info("Health risk model loaded successfully", model_path=str(self.model_path))
                
                # The model is a Pipeline, so we can check its structure
                if hasattr(self.model, 'named_steps'):
                    logger.info(f"Model pipeline steps: {list(self.model.named_steps.keys())}")
                    
                    # Get the preprocessor to understand what it expects
                    if 'preprocessor' in self.model.named_steps:
                        preprocessor = self.model.named_steps['preprocessor']
                        if hasattr(preprocessor, 'transformers'):
                            logger.info("Preprocessor transformers found")
                
                return True
            else:
                # If primary model is not found, try loading fallback models
                for model_name, path in self.fallback_models.items():
                    if path.exists():
                        self.model = joblib.load(path)
                        self.model_path = path
                        self.is_loaded = True
                        logger.info(f"Fallback health risk model ({model_name}) loaded", model_path=str(path))
                        return True
                
                logger.error("No health risk model found", 
                           primary_path=str(self.model_path),
                           fallback_paths=[str(p) for p in self.fallback_models.values()])
                return False
                
        except Exception as e:
            logger.error("Failed to load health risk model", error=str(e), path_attempted=str(self.model_path))
            return False

    def _estimate_symptoms_from_risk_factors(self, input_data: Dict[str, Any]) -> Dict[str, int]:
        """Estimate symptom probabilities based on environmental risk factors"""
        water_quality = input_data.get('water_quality_data', {})
        age = input_data.get('age', 30)
        sanitation = input_data.get('sanitation', 'improved')
        
        # Base symptom probabilities (all start at 0)
        symptoms = {
            'diarrhea': 0,
            'vomiting': 0,
            'fever': 0,
            'fatigue': 0,
            'jaundice': 0,
            'headache': 0,
            'loss_appetite': 0,
            'muscle_aches': 0
        }
        
        # Calculate risk score based on water quality and other factors
        risk_score = 0.0
        
        # Water quality risk factors
        ph = water_quality.get('ph_level', 7.0)
        if ph < 6.5 or ph > 8.5:
            risk_score += 0.3
        
        turbidity = water_quality.get('turbidity', 1.0)
        if turbidity > 4:
            risk_score += 0.4
        elif turbidity > 2:
            risk_score += 0.2
        
        chloramines = water_quality.get('chloramines', 0)
        if chloramines > 4:
            risk_score += 0.2
        
        # Sanitation risk factors
        sanitation_lower = sanitation.lower()
        if 'open' in sanitation_lower or 'defecation' in sanitation_lower:
            risk_score += 0.5
        elif 'unimproved' in sanitation_lower:
            risk_score += 0.3
        
        # Water source risk factors
        water_source = input_data.get('water_source', '').lower()
        if any(risk_source in water_source for risk_source in ['surface', 'river', 'pond']):
            risk_score += 0.3
        elif 'well' in water_source:
            risk_score += 0.1
        
        # Age vulnerability
        if age < 5:
            risk_score += 0.3
        elif age > 65:
            risk_score += 0.2
        
        # Convert risk score to symptom predictions
        if risk_score > 0.7:  # High risk
            symptoms.update({
                'diarrhea': 1,
                'vomiting': 1,
                'fever': 1,
                'fatigue': 1,
                'loss_appetite': 1
            })
        elif risk_score > 0.4:  # Medium risk
            symptoms.update({
                'diarrhea': 1,
                'fatigue': 1,
                'headache': 1
            })
        elif risk_score > 0.2:  # Low-medium risk
            symptoms.update({
                'fatigue': 1
            })
        
        return symptoms

    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for the health model pipeline"""
        try:
            # Get basic data - keep as strings/original types for the pipeline
            age = int(input_data.get('age', 30))
            sex = str(input_data.get('sex', 'male')).lower()
            sanitation = str(input_data.get('sanitation', 'improved')).lower()
            water_source = str(input_data.get('water_source', 'piped_water')).lower()
            
            # Normalize categorical values to match training data format
            # Based on common patterns, simplify water source
            if 'piped' in water_source or 'tap' in water_source:
                water_source = 'piped_water'
            elif 'well' in water_source or 'ground' in water_source or 'borehole' in water_source:
                water_source = 'well_water'
            elif 'surface' in water_source or 'river' in water_source or 'pond' in water_source:
                water_source = 'surface_water'
            elif 'spring' in water_source:
                water_source = 'spring_water'
            else:
                water_source = 'piped_water'  # default
            
            # Normalize sanitation
            if 'open' in sanitation or 'defecation' in sanitation:
                sanitation = 'open_defecation'
            elif 'unimproved' in sanitation:
                sanitation = 'unimproved'
            else:
                sanitation = 'improved'
            
            # Normalize sex
            if 'f' in sex.lower():
                sex = 'female'
            else:
                sex = 'male'
            
            # Estimate symptoms based on risk factors
            symptoms = self._estimate_symptoms_from_risk_factors(input_data)
            
            # Create the raw dataframe that the pipeline expects
            # IMPORTANT: Use the exact column order and types that the model was trained with
            data = {
                'sex': sex,  # Keep as string
                'sanitation': sanitation,  # Keep as string  
                'water_source': water_source,  # Keep as string
                'age': age,  # Integer
                'diarrhea': symptoms['diarrhea'],  # Integer (0 or 1)
                'vomiting': symptoms['vomiting'],
                'fever': symptoms['fever'],
                'fatigue': symptoms['fatigue'],
                'jaundice': symptoms['jaundice'],
                'headache': symptoms['headache'],
                'loss_appetite': symptoms['loss_appetite'],
                'muscle_aches': symptoms['muscle_aches']
            }
            
            # Create DataFrame with the exact column order the model expects
            df = pd.DataFrame([data])
            
            # Ensure column order matches what the pipeline expects
            # The training script shows: X = df.drop(columns=["true_prob"])
            # So we need the columns in the same order as the training data
            expected_cols = ['age', 'sex', 'sanitation', 'water_source',
                           'diarrhea', 'vomiting', 'fever', 'fatigue',
                           'jaundice', 'headache', 'loss_appetite', 'muscle_aches']
            
            # Reorder columns to match expected order
            df = df[expected_cols]
            
            logger.info("Preprocessed health input for pipeline", 
                       shape=df.shape,
                       columns=list(df.columns),
                       sex=df['sex'].iloc[0],
                       sanitation=df['sanitation'].iloc[0],
                       water_source=df['water_source'].iloc[0],
                       age=df['age'].iloc[0],
                       symptom_count=sum([symptoms[k] for k in symptoms]))
            
            return df
            
        except Exception as e:
            logger.error("Error preprocessing health risk input", error=str(e), exc_info=True)
            raise
    
    def predict_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict health risk probability"""
        if not self.is_loaded:
            if not self.load_model():
                logger.warning("Health model unavailable, using rule-based fallback")
                return self._rule_based_prediction(input_data)
        
        try:
            # Preprocess input - the pipeline will handle the rest
            X = self.preprocess_input(input_data)
            
            logger.info("Making health prediction with pipeline model", 
                       input_shape=X.shape,
                       columns=list(X.columns),
                       model_type=type(self.model).__name__,
                       has_pipeline='Pipeline' in str(type(self.model)))
            
            # Make prediction using the pipeline (which includes preprocessing)
            # The pipeline expects raw data and will handle one-hot encoding internally
            risk_prob = float(self.model.predict(X)[0])
            
            # The model outputs a probability between 0 and 1
            risk_score = max(0.0, min(1.0, risk_prob))
            
            # Determine risk level
            if risk_score < 0.3:
                risk_level = "LOW"
                risk_color = "GREEN"
            elif risk_score < 0.7:
                risk_level = "MEDIUM"
                risk_color = "YELLOW"
            else:
                risk_level = "HIGH"
                risk_color = "RED"
            
            # Generate additional insights
            risk_factors = self._identify_health_risk_factors(input_data, X)
            predicted_diseases = self._predict_diseases(risk_score, input_data)
            
            result = {
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "confidence": 0.85,
                "predicted_diseases": predicted_diseases,
                "primary_risk_factors": risk_factors,
                "population_at_risk": self._estimate_population_at_risk(input_data, risk_score),
                "model_version": f"health_ml_{self.model_path.stem}_pipeline",
                "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                "estimated_symptoms": self._get_estimated_symptoms(X)
            }
            
            logger.info("Health prediction completed successfully", 
                       risk_score=risk_score, 
                       risk_level=risk_level,
                       model_used=self.model_path.stem)
            
            return result
            
        except Exception as e:
            logger.error("Health risk ML prediction failed", 
                        error=str(e),
                        exc_info=True)
            logger.warning("Falling back to rule-based prediction")
            return self._rule_based_prediction(input_data)
    
    def _get_estimated_symptoms(self, X: pd.DataFrame) -> Dict[str, bool]:
        """Extract estimated symptoms from the preprocessed data"""
        symptom_cols = ['diarrhea', 'vomiting', 'fever', 'fatigue', 
                       'jaundice', 'headache', 'loss_appetite', 'muscle_aches']
        
        symptoms = {}
        for col in symptom_cols:
            if col in X.columns:
                symptoms[col] = bool(X[col].iloc[0])
        
        return symptoms
    
    def _rule_based_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced rule-based fallback prediction"""
        water_quality = input_data.get('water_quality_data', {})
        
        risk_score = 0.3  # Base risk
        risk_factors = []
        
        # Age vulnerability
        age = input_data.get('age', 30)
        if age < 5:
            risk_score += 0.25
            risk_factors.append("Very young age (high vulnerability)")
        elif age > 65:
            risk_score += 0.2
            risk_factors.append("Advanced age (increased vulnerability)")
        
        # Water quality factors
        ph = water_quality.get('ph_level', 7.0)
        if ph < 6.5 or ph > 8.5:
            risk_score += 0.2
            risk_factors.append("Unsafe water pH levels")
        
        turbidity = water_quality.get('turbidity', 1.0)
        if turbidity > 4:
            risk_score += 0.25
            risk_factors.append("Very high water turbidity")
        elif turbidity > 2:
            risk_score += 0.15
            risk_factors.append("Elevated water turbidity")
        
        # Chemical contaminants
        chloramines = water_quality.get('chloramines', 0)
        if chloramines > 4:
            risk_score += 0.15
            risk_factors.append("High disinfection byproducts")
        
        # Sanitation
        sanitation = input_data.get('sanitation', 'improved')
        if sanitation == 'open_defecation':
            risk_score += 0.3
            risk_factors.append("Open defecation practices")
        elif sanitation == 'unimproved':
            risk_score += 0.2
            risk_factors.append("Unimproved sanitation facilities")
        
        # Water source risk
        water_source = input_data.get('water_source', 'piped_water').lower()
        if any(risk_source in water_source for risk_source in ['surface', 'river', 'pond']):
            risk_score += 0.2
            risk_factors.append("High-risk water source")
        elif 'well' in water_source:
            risk_score += 0.1
            risk_factors.append("Potential groundwater contamination")
        
        risk_score = min(1.0, risk_score)
        
        if risk_score < 0.3:
            risk_level = "LOW"
            risk_color = "GREEN"
        elif risk_score < 0.7:
            risk_level = "MEDIUM"
            risk_color = "YELLOW"
        else:
            risk_level = "HIGH"
            risk_color = "RED"
        
        if not risk_factors:
            risk_factors = ["General environmental factors"]
        
        return {
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "confidence": 0.75,
            "predicted_diseases": self._predict_diseases(risk_score, input_data),
            "primary_risk_factors": risk_factors,
            "population_at_risk": self._estimate_population_at_risk(input_data, risk_score),
            "model_version": "rule_based_fallback_v1.2",
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "Using rule-based prediction (ML model unavailable or failed)"
        }
    
    def _identify_health_risk_factors(self, original_data: Dict[str, Any], processed_data: pd.DataFrame) -> List[str]:
        """Identify main health risk factors"""
        factors = []
        
        water_quality = original_data.get('water_quality_data', {})
        age = original_data.get('age', 30)
        sanitation = original_data.get('sanitation', 'improved')
        
        # Age factors
        if age < 5:
            factors.append("Very young age (high vulnerability)")
        elif age > 65:
            factors.append("Advanced age (increased vulnerability)")
        
        # Symptom-based factors
        symptom_cols = ['diarrhea', 'vomiting', 'fever', 'fatigue', 'headache']
        active_symptoms = []
        for col in symptom_cols:
            if col in processed_data.columns and processed_data[col].iloc[0] == 1:
                active_symptoms.append(col)
        
        if active_symptoms:
            factors.append(f"Risk indicators: {', '.join(active_symptoms)}")
        
        # Environmental factors
        ph = water_quality.get('ph_level', 7.0)
        if ph < 6.5 or ph > 8.5:
            factors.append("Unsafe water pH levels")
        
        turbidity = water_quality.get('turbidity', 0)
        if turbidity > 4:
            factors.append("Very high water turbidity")
        
        if sanitation in ['open_defecation', 'unimproved']:
            factors.append("Poor sanitation infrastructure")
        
        if not factors:
            factors.append("Environmental and demographic factors")
        
        return factors[:5]
    
    def _predict_diseases(self, risk_score: float, input_data: Dict[str, Any]) -> List[str]:
        """Predict likely waterborne diseases"""
        diseases = []
        water_quality = input_data.get('water_quality_data', {})
        age = input_data.get('age', 30)
        sanitation = input_data.get('sanitation', 'improved')
        
        if risk_score > 0.7:
            diseases.extend(["acute_diarrhea", "gastroenteritis", "dehydration"])
            if sanitation in ['open_defecation', 'unimproved']:
                diseases.append("cholera")
        elif risk_score > 0.5:
            diseases.extend(["gastroenteritis", "mild_diarrhea"])
        elif risk_score > 0.3:
            diseases.append("gastroenteritis")
        else:
            diseases.append("general_gi_discomfort")
        
        if age < 5 and risk_score > 0.4:
            diseases.append("severe_dehydration")
        elif age > 65 and risk_score > 0.5:
            diseases.append("prolonged_illness")
        
        ph = water_quality.get('ph_level', 7)
        if ph < 6.5 and risk_score > 0.4:
            diseases.append("dysentery")
        
        turbidity = water_quality.get('turbidity', 1)
        if turbidity > 4 and risk_score > 0.5:
            diseases.append("bacterial_infection")
        
        unique_diseases = list(dict.fromkeys(diseases))
        return unique_diseases[:4]
    
    def _estimate_population_at_risk(self, input_data: Dict[str, Any], risk_score: float) -> int:
        """Estimate population at risk"""
        base_population = input_data.get('population_density', 250) * 0.5
        
        if risk_score > 0.7:
            risk_multiplier = 0.8
        elif risk_score > 0.5:
            risk_multiplier = 0.6
        elif risk_score > 0.3:
            risk_multiplier = 0.4
        else:
            risk_multiplier = 0.2
        
        age = input_data.get('age', 30)
        if age < 5 or age > 65:
            vulnerability_multiplier = 1.5
        else:
            vulnerability_multiplier = 1.0
        
        sanitation = input_data.get('sanitation', 'improved')
        if sanitation == 'open_defecation':
            sanitation_multiplier = 2.0
        elif sanitation == 'unimproved':
            sanitation_multiplier = 1.5
        else:
            sanitation_multiplier = 1.0

        # --- IMPROVEMENT ---
        # Added logging for multipliers to assist with debugging complex cases.
        logger.info(
            "Estimating population at risk",
            risk_multiplier=risk_multiplier,
            vulnerability_multiplier=vulnerability_multiplier,
            sanitation_multiplier=sanitation_multiplier,
            base_population=base_population
        )
        
        total_at_risk = int(base_population * risk_multiplier * vulnerability_multiplier * sanitation_multiplier)
        return max(1, min(total_at_risk, int(base_population * 2)))


class DiseasePredictor:
    """Main disease prediction orchestrator"""
    
    def __init__(self):
        self.water_model = WaterQualityModel()
        self.health_model = HealthRiskModel()
        self.version = "v1.1" # Version update
        self.feature_columns = [
            'ph_level', 'turbidity', 'chloramines', 'sulfate',
            'conductivity', 'hardness', 'solids', 'organic_carbon',
            'trihalomethanes', 'age', 'population_density'
        ]
        
        # --- IMPROVEMENT ---
        # Made the combined risk weights configurable class attributes.
        self.water_risk_weight = 0.6
        self.health_risk_weight = 0.4
        
    def load_models(self) -> bool:
        """Load both models"""
        water_loaded = self.water_model.load_model()
        health_loaded = self.health_model.load_model()
        
        if not water_loaded and not health_loaded:
            logger.error("No ML models could be loaded")
            return False
        
        logger.info(f"Models loaded - Water: {water_loaded}, Health: {health_loaded}")
        return True
    
    def predict_water_quality_risk(self, water_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict water quality risk"""
        # The input for water quality model is the nested water_quality_data dictionary
        # plus some root-level fields.
        combined_water_input = water_data.get('water_quality_data', {})
        for key in ['water_source', 'sanitation', 'population_density']:
            if key in water_data:
                combined_water_input[key] = water_data[key]
        return self.water_model.predict_risk(combined_water_input)
    
    def predict_health_risk(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict health risk"""
        # The health model expects the entire top-level dictionary
        return self.health_model.predict_risk(health_data)
    
    def predict_combined_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict combined water quality and health risk"""
        try:
            results = {
                "water_quality_risk": None,
                "health_risk": None,
                "combined_risk": None,
                "recommendations": [],
                "assessment_type": "combined_risk_assessment",
                "models_used": []
            }
            
            # --- Water quality prediction ---
            try:
                # --- IMPROVEMENT ---
                # Made data extraction more robust to handle cases where 'water_quality_data'
                # might be None instead of a dictionary.
                water_input = input_data.get('water_quality_data') or {}
                
                # Update with relevant top-level fields
                water_input.update({
                    'water_source': input_data.get('water_source'),
                    'sanitation': input_data.get('sanitation'),
                    'population_density': input_data.get('population_density')
                })
                
                results["water_quality_risk"] = self.water_model.predict_risk(water_input)
                if "rule_based" not in results["water_quality_risk"].get("model_version", ""):
                     results["models_used"].append("water_quality_rf_model")

            except Exception as e:
                logger.error("Water quality prediction failed during combined assessment", error=str(e), exc_info=True)
            
            # --- Health risk prediction ---
            try:
                # <<< CRITICAL FIX: Pass the *entire* input_data dictionary to the health model
                # This ensures it has access to the nested 'water_quality_data'
                results["health_risk"] = self.health_model.predict_risk(input_data)
                if "rule_based" not in results["health_risk"].get("model_version", ""):
                     results["models_used"].append("health_risk_rf_model")

            except Exception as e:
                logger.error("Health risk prediction failed during combined assessment", error=str(e), exc_info=True)
            
            # --- Combined risk calculation ---
            if results["water_quality_risk"] and results["health_risk"]:
                water_risk = results["water_quality_risk"]["risk_score"]
                health_risk = results["health_risk"]["risk_score"]
                
                # --- IMPROVEMENT ---
                # Using configurable weights for the weighted average calculation.
                combined_score = (water_risk * self.water_risk_weight) + (health_risk * self.health_risk_weight)
                
                if combined_score < 0.3:
                    risk_level = "LOW"
                    risk_color = "GREEN"
                elif combined_score < 0.7:
                    risk_level = "MEDIUM"
                    risk_color = "YELLOW"
                else:
                    risk_level = "HIGH"
                    risk_color = "RED"
                
                # The logic for combined_risk was already complete, ensuring it's properly assigned.
                results["combined_risk"] = {
                    "risk_score": float(combined_score),
                    "risk_level": risk_level,
                    "risk_color": risk_color,
                    "confidence": min(
                        results["water_quality_risk"].get("confidence", 0.8),
                        results["health_risk"].get("confidence", 0.8)
                    )
                }
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)
            
            return convert_to_serializable(results)
            
        except Exception as e:
            logger.error("Combined risk prediction failed", error=str(e), exc_info=True)
            # Ensure a fallback response is provided
            return {
                "error": "Failed to compute combined risk",
                "details": str(e)
            }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on predictions"""
        recommendations = []
        
        water_risk = results.get("water_quality_risk")
        health_risk = results.get("health_risk")
        
        if water_risk and water_risk["risk_level"] == "HIGH":
            recommendations.append("High water contamination risk detected. Advise boiling water before consumption.")
            recommendations.append("Immediate inspection of water source and distribution network is required.")
        
        if health_risk and health_risk["risk_level"] == "HIGH":
            recommendations.append("High health risk predicted. Increase public health surveillance for waterborne diseases.")
            recommendations.append("Prepare local clinics for a potential increase in gastrointestinal cases.")
        
        if results.get("combined_risk", {}).get("risk_level") == "HIGH":
            recommendations.append("CRITICAL ALERT: Issue an immediate public health advisory for the affected area.")
        
        if water_risk and water_risk["risk_level"] == "MEDIUM":
            recommendations.append("Water quality is suboptimal. Recommend point-of-use water filters for vulnerable populations.")

        if health_risk and health_risk["risk_level"] == "MEDIUM":
             recommendations.append("Moderate health risk. Launch a public awareness campaign on safe hygiene practices.")

        if not recommendations:
            recommendations.append("Continue routine monitoring of water quality and public health indicators.")
        
        return list(dict.fromkeys(recommendations)) # Return unique recommendations
    
    def train_model(self, training_data: List[Dict], labels: List[float]) -> Dict[str, Any]:
        """Placeholder for model retraining"""
        return {
            "status": "training_not_supported",
            "message": "Model retraining requires specialized infrastructure",
            "suggestion": "Contact ML team for model updates"
        }

# Initialize global model instance
disease_prediction_model = DiseasePredictor()


def get_model() -> DiseasePredictor:
    """Get the global model instance"""
    global disease_prediction_model
    if not (disease_prediction_model.water_model.is_loaded or disease_prediction_model.health_model.is_loaded):
        disease_prediction_model.load_models()
    return disease_prediction_model


# Model initialization function
def initialize_models():
    """Initialize ML models on startup"""
    try:
        success = disease_prediction_model.load_models()
        if success:
            logger.info("ML models initialized successfully")
        else:
            logger.warning("Some ML models failed to load")
        return success
    except Exception as e:
        logger.error("ML model initialization failed", error=str(e))
        return False
    

RiskPredictor = DiseasePredictor