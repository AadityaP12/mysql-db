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

print("🔎 MODEL_PATH:", settings.MODEL_PATH)
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
        self.feature_columns = [
            'ph', 'hardness', 'solids', 'chloramines', 'sulfate',
            'conductivity', 'organic_carbon', 'trihalomethanes',
            'turbidity', 'source_type', 'season', 'month'
        ]
        self.required_features = ['ph', 'turbidity', 'source_type']
        
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
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for the model"""
        try:
            # Create a dictionary with default values
            processed_data = {
                'ph': input_data.get('ph_level', 7.0),
                'hardness': input_data.get('hardness', 200.0),
                'solids': input_data.get('solids', 20000.0),
                'chloramines': input_data.get('chloramines', 7.0),
                'sulfate': input_data.get('sulfate', 300.0),
                'conductivity': input_data.get('conductivity', 400.0),
                'organic_carbon': input_data.get('organic_carbon', 14.0),
                'trihalomethanes': input_data.get('trihalomethanes', 80.0),
                'turbidity': input_data.get('turbidity', 3.0),
                'source_type': input_data.get('source_type', 'surface'),
                'season': input_data.get('season', 'summer'),
                'month': input_data.get('month', 6)
            }
            
            # Convert to DataFrame
            df = pd.DataFrame([processed_data])
            
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in df.columns:
                    # Set reasonable defaults for missing columns
                    if col in ['source_type']:
                        df[col] = 'surface'
                    elif col in ['season']:
                        df[col] = 'summer'
                    elif col in ['month']:
                        df[col] = 6
                    else:
                        df[col] = 0.0
            
            return df[self.feature_columns]
            
        except Exception as e:
            logger.error("Error preprocessing water quality input", error=str(e))
            raise
    
    def predict_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict water quality risk"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Water quality model not available")
        
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Make prediction (returns risk percentage 0-100)
            risk_percent = self.model.predict(X)[0]
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
                "confidence": 0.85,  # Model confidence
                "primary_risk_factors": risk_factors,
                "model_version": "rf_water_v1.0",
                "prediction_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error("Water quality prediction failed", error=str(e))
            raise
    
    def _identify_risk_factors(self, original_data: Dict[str, Any], processed_data: pd.DataFrame) -> List[str]:
        """Identify main risk factors based on input values"""
        factors = []
        
        ph = original_data.get('ph_level', 7.0)
        turbidity = original_data.get('turbidity', 0.0)
        chloramines = original_data.get('chloramines', 0.0)
        
        if ph < 6.5 or ph > 8.5:
            factors.append("pH levels outside safe range")
        
        if turbidity > 4.0:
            factors.append("High turbidity levels")
        
        if chloramines > 4.0:
            factors.append("Elevated chloramine levels")
        
        if original_data.get('source_type') == 'groundwater':
            factors.append("Groundwater source risks")
        
        if not factors:
            factors.append("Multiple water quality parameters")
        
        return factors


class HealthRiskModel:
    """Health Risk Prediction Model"""
    
    def __init__(self):
        self.model = None
        self.model_path = Path(settings.MODEL_PATH) / "rf_health_model.pkl"
        self.fallback_models = {
            "xgb": Path(settings.MODEL_PATH) / "xgb_health_model.pkl",
            "lgb": Path(settings.MODEL_PATH) / "lgb_health_model.pkl"
        }
        self.is_loaded = False
        self.feature_columns = [
            'age', 'sex', 'income', 'sanitation', 'water_source',
            'ph', 'hardness', 'solids', 'chloramines', 'sulfate',
            'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity'
        ]
        self.required_features = ['age', 'sex', 'water_source']
    
    def load_model(self) -> bool:
        """Load the trained health risk model"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                logger.info("Health risk model loaded successfully", model_path=str(self.model_path))
                return True
            else:
                # Try loading fallback models
                for model_name, path in self.fallback_models.items():
                    if path.exists():
                        self.model = joblib.load(path)
                        self.is_loaded = True
                        logger.info(f"Fallback health risk model ({model_name}) loaded", model_path=str(path))
                        return True
                
                logger.error("No health risk model found",
                           primary_path=str(self.model_path),
                           fallback_paths=[str(p) for p in self.fallback_models.values()])
                return False
                
        except Exception as e:
            logger.error("Failed to load health risk model", error=str(e))
            return False
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for the health model"""
        try:
            # Extract water quality data
            water_quality = input_data.get('water_quality_data', {})
            
            processed_data = {
                'age': input_data.get('age', 30),
                'sex': input_data.get('sex', 'male'),
                'income': input_data.get('income', 50000),
                'sanitation': input_data.get('sanitation', 'improved'),
                'water_source': input_data.get('water_source', 'piped'),
                'ph': water_quality.get('ph_level', 7.0),
                'hardness': water_quality.get('hardness', 200.0),
                'solids': water_quality.get('solids', 20000.0),
                'chloramines': water_quality.get('chloramines', 7.0),
                'sulfate': water_quality.get('sulfate', 300.0),
                'conductivity': water_quality.get('conductivity', 400.0),
                'organic_carbon': water_quality.get('organic_carbon', 14.0),
                'trihalomethanes': water_quality.get('trihalomethanes', 80.0),
                'turbidity': water_quality.get('turbidity', 3.0)
            }
            
            # Convert to DataFrame
            df = pd.DataFrame([processed_data])
            
            # Ensure all required columns are present with defaults
            for col in self.feature_columns:
                if col not in df.columns:
                    if col == 'sex':
                        df[col] = 'male'
                    elif col == 'sanitation':
                        df[col] = 'improved'
                    elif col == 'water_source':
                        df[col] = 'piped'
                    elif col == 'age':
                        df[col] = 30
                    elif col == 'income':
                        df[col] = 50000
                    else:
                        df[col] = 0.0
            
            return df[self.feature_columns]
            
        except Exception as e:
            logger.error("Error preprocessing health risk input", error=str(e))
            raise
    
    def predict_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict health risk probability"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Health risk model not available")
        
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Make prediction (returns probability 0-1)
            risk_prob = self.model.predict(X)[0]
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
            
            # Identify risk factors
            risk_factors = self._identify_health_risk_factors(input_data, X)
            
            # Predict likely diseases based on risk score and factors
            predicted_diseases = self._predict_diseases(risk_score, input_data)
            
            return {
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "confidence": 0.82,  # Model confidence
                "predicted_diseases": predicted_diseases,
                "primary_risk_factors": risk_factors,
                "population_at_risk": self._estimate_population_at_risk(input_data, risk_score),
                "model_version": "rf_health_v1.0",
                "prediction_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error("Health risk prediction failed", error=str(e))
            raise
    
    def _identify_health_risk_factors(self, original_data: Dict[str, Any], processed_data: pd.DataFrame) -> List[str]:
        """Identify main health risk factors"""
        factors = []
        
        water_quality = original_data.get('water_quality_data', {})
        age = original_data.get('age', 30)
        sanitation = original_data.get('sanitation', 'improved')
        
        # Age-based risk
        if age < 5 or age > 65:
            factors.append("Age-related vulnerability")
        
        # Water quality risks
        ph = water_quality.get('ph_level', 7.0)
        if ph < 6.5 or ph > 8.5:
            factors.append("Poor water pH levels")
        
        turbidity = water_quality.get('turbidity', 0)
        if turbidity > 4:
            factors.append("High water turbidity")
        
        # Sanitation risks
        if sanitation in ['unimproved', 'open_defecation']:
            factors.append("Poor sanitation facilities")
        
        # Chloramine levels
        chloramines = water_quality.get('chloramines', 0)
        if chloramines > 4:
            factors.append("Elevated disinfection byproducts")
        
        if not factors:
            factors.append("Environmental and water quality factors")
        
        return factors
    
    def _predict_diseases(self, risk_score: float, input_data: Dict[str, Any]) -> List[str]:
        """Predict likely waterborne diseases based on risk score"""
        diseases = []
        
        if risk_score > 0.7:
            diseases.extend(["diarrhea", "gastroenteritis", "cholera"])
        elif risk_score > 0.4:
            diseases.extend(["diarrhea", "gastroenteritis"])
        elif risk_score > 0.2:
            diseases.append("gastroenteritis")
        else:
            diseases.append("general_waterborne")
        
        # Add specific diseases based on water quality
        water_quality = input_data.get('water_quality_data', {})
        if water_quality.get('ph_level', 7) < 6.5:
            if "dysentery" not in diseases:
                diseases.append("dysentery")
        
        return diseases[:3]  # Return top 3 most likely
    
    def _estimate_population_at_risk(self, input_data: Dict[str, Any], risk_score: float) -> int:
        """Estimate population at risk in the area"""
        # This is a simplified estimation
        base_population = input_data.get('population_density', 250) * 0.5  # Assume 0.5 km radius
        
        if risk_score > 0.7:
            return int(base_population * 0.8)
        elif risk_score > 0.4:
            return int(base_population * 0.5)
        elif risk_score > 0.2:
            return int(base_population * 0.3)
        else:
            return int(base_population * 0.1)


class DiseasePredictor:
    """Main disease prediction orchestrator"""
    
    def __init__(self):
        self.water_model = WaterQualityModel()
        self.health_model = HealthRiskModel()
        self.version = "v1.0"
        self.feature_columns = [
            'ph_level', 'turbidity', 'chloramines', 'sulfate',
            'conductivity', 'hardness', 'solids', 'organic_carbon',
            'trihalomethanes', 'age', 'population_density'
        ]
        
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
        return self.water_model.predict_risk(water_data)
    
    def predict_health_risk(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict health risk"""
        return self.health_model.predict_risk(health_data)
    
    def predict_combined_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict combined water quality and health risk"""
        try:
            results = {
                "water_quality_risk": None,
                "health_risk": None,
                "combined_risk": None,
                "recommendations": []
            }
            
            # Water quality prediction
            if self.water_model.is_loaded:
                try:
                    results["water_quality_risk"] = self.water_model.predict_risk(input_data)
                except Exception as e:
                    logger.error("Water quality prediction failed", error=str(e))
            
            # Health risk prediction
            if self.health_model.is_loaded:
                try:
                    results["health_risk"] = self.health_model.predict_risk(input_data)
                except Exception as e:
                    logger.error("Health risk prediction failed", error=str(e))
            
            # Combined risk calculation
            if results["water_quality_risk"] and results["health_risk"]:
                water_risk = results["water_quality_risk"]["risk_score"]
                health_risk = results["health_risk"]["risk_score"]
                
                # Weighted average (water quality has slightly more weight)
                combined_score = (water_risk * 0.6) + (health_risk * 0.4)
                
                if combined_score < 0.3:
                    risk_level = "LOW"
                    risk_color = "GREEN"
                elif combined_score < 0.7:
                    risk_level = "MEDIUM"
                    risk_color = "YELLOW"
                else:
                    risk_level = "HIGH"
                    risk_color = "RED"
                
                results["combined_risk"] = {
                    "risk_score": float(combined_score),
                    "risk_level": risk_level,
                    "risk_color": risk_color,
                    "confidence": 0.83
                }
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)
            
            return convert_to_serializable(results)
            
        except Exception as e:
            logger.error("Combined risk prediction failed", error=str(e))
            raise
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on predictions"""
        recommendations = []
        
        water_risk = results.get("water_quality_risk")
        health_risk = results.get("health_risk")
        
        if water_risk and water_risk["risk_level"] == "HIGH":
            recommendations.append("Boil water before consumption")
            recommendations.append("Implement water treatment measures")
        
        if health_risk and health_risk["risk_level"] == "HIGH":
            recommendations.append("Increase health monitoring in the area")
            recommendations.append("Distribute oral rehydration salts")
        
        if results.get("combined_risk", {}).get("risk_level") == "HIGH":
            recommendations.append("Issue immediate public health advisory")
            recommendations.append("Deploy rapid response health team")
        
        if not recommendations:
            recommendations.append("Continue routine monitoring")
        
        return recommendations
    
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