import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Dict, List, Any, Optional, Tuple
import os
import logging
from datetime import datetime, timezone
import json

from app.config import settings

logger = logging.getLogger(__name__)


class WaterborneDiseasePredictionModel:
    """
    Machine Learning model for predicting waterborne disease outbreaks
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(settings.MODEL_PATH, "waterborne_disease_model.joblib")
        self.scaler_path = os.path.join(settings.MODEL_PATH, "feature_scaler.joblib")
        self.encoder_path = os.path.join(settings.MODEL_PATH, "label_encoder.joblib")
        
        # Model components
        self.risk_model = None  # For risk score prediction (regression)
        self.outbreak_model = None  # For outbreak classification
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Feature columns expected by the model
        self.feature_columns = [
            'ph_level', 'turbidity', 'residual_chlorine', 'temperature',
            'population_density', 'symptom_count', 'sanitation_score',
            'rainfall_mm', 'humidity_percent', 'season_encoded',
            'water_source_encoded', 'previous_outbreak_count'
        ]
        
        # Load pre-trained model if available
        self.load_model()
    
    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare and transform features for prediction
        """
        try:
            # Extract water quality data
            water_quality = data.get('water_quality_data', {})
            
            # Basic water quality features
            features = {
                'ph_level': water_quality.get('ph_level', 7.0),
                'turbidity': water_quality.get('turbidity', 0.0),
                'residual_chlorine': water_quality.get('residual_chlorine', 0.0),
                'temperature': water_quality.get('temperature', 25.0),
            }
            
            # Population and health features
            features['population_density'] = data.get('population_density', 100)
            features['symptom_count'] = sum(data.get('health_symptoms_count', {}).values())
            features['sanitation_score'] = data.get('sanitation_score', 5.0)
            
            # Environmental features (mock data - would come from weather API)
            seasonal_factors = data.get('seasonal_factors', {})
            features['rainfall_mm'] = seasonal_factors.get('rainfall_mm', 50.0)
            features['humidity_percent'] = seasonal_factors.get('humidity_percent', 70.0)
            
            # Encode categorical features
            current_month = datetime.now().month
            if current_month in [6, 7, 8, 9]:  # Monsoon season
                features['season_encoded'] = 1
            elif current_month in [12, 1, 2]:  # Winter
                features['season_encoded'] = 0
            else:  # Other seasons
                features['season_encoded'] = 2
            
            # Water source encoding (simplified)
            water_source_map = {
                'tube_well': 0, 'hand_pump': 1, 'dug_well': 2,
                'spring': 3, 'surface_water': 4, 'piped_water': 5
            }
            water_source = data.get('water_source', 'tube_well')
            features['water_source_encoded'] = water_source_map.get(water_source, 0)
            
            # Historical outbreak data
            historical_data = data.get('historical_outbreak_data', [])
            features['previous_outbreak_count'] = len(historical_data)
            
            # Convert to DataFrame for consistent processing
            feature_df = pd.DataFrame([features])
            
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0
            
            # Select and order features
            feature_df = feature_df[self.feature_columns]
            
            # Scale features if scaler is available
            if hasattr(self.scaler, 'transform'):
                features_scaled = self.scaler.transform(feature_df)
            else:
                features_scaled = feature_df.values
            
            return features_scaled[0]  # Return single sample
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            # Return default features
            return np.zeros(len(self.feature_columns))
    
    def predict_risk_score(self, features: np.ndarray) -> float:
        """
        Predict risk score (0-1) for waterborne disease outbreak
        """
        try:
            if self.risk_model is None:
                # Use rule-based fallback if model not available
                return self._rule_based_risk_assessment(features)
            
            risk_score = self.risk_model.predict([features])[0]
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error predicting risk score: {str(e)}")
            return 0.5  # Default moderate risk
    
    def predict_outbreak_probability(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict probability of disease outbreak and classify diseases
        """
        try:
            if self.outbreak_model is None:
                return self._rule_based_outbreak_assessment(features)
            
            # Get prediction probabilities
            probabilities = self.outbreak_model.predict_proba([features])[0]
            prediction = self.outbreak_model.predict([features])[0]
            
            # Map to disease types (would be defined during training)
            disease_classes = ['no_outbreak', 'diarrhea', 'cholera', 'typhoid', 'hepatitis_a']
            
            result = {
                'outbreak_probability': 1.0 - probabilities[0],  # Probability of any outbreak
                'predicted_diseases': [],
                'disease_probabilities': {}
            }
            
            # Get top disease predictions
            for i, prob in enumerate(probabilities[1:], 1):  # Skip 'no_outbreak'
                if prob > 0.2:  # Threshold for considering a disease
                    disease = disease_classes[i]
                    result['predicted_diseases'].append(disease)
                    result['disease_probabilities'][disease] = prob
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting outbreak: {str(e)}")
            return {
                'outbreak_probability': 0.3,
                'predicted_diseases': ['general_waterborne'],
                'disease_probabilities': {'general_waterborne': 0.3}
            }
    
    def _rule_based_risk_assessment(self, features: np.ndarray) -> float:
        """
        Rule-based risk assessment fallback when ML model is not available
        """
        try:
            # Map features back to interpretable values
            ph_level = features[0] if len(features) > 0 else 7.0
            turbidity = features[1] if len(features) > 1 else 0.0
            chlorine = features[2] if len(features) > 2 else 0.0
            symptom_count = features[5] if len(features) > 5 else 0
            sanitation_score = features[6] if len(features) > 6 else 5.0
            
            risk_score = 0.0
            
            # Water quality risks
            if ph_level < 6.5 or ph_level > 8.5:
                risk_score += 0.3
            if turbidity > 5.0:
                risk_score += 0.25
            if chlorine < 0.2:
                risk_score += 0.2
            
            # Health indicators
            if symptom_count > 5:
                risk_score += 0.4
            elif symptom_count > 2:
                risk_score += 0.2
            
            # Sanitation factors
            if sanitation_score < 3:
                risk_score += 0.3
            elif sanitation_score < 5:
                risk_score += 0.15
            
            # Seasonal factors (simplified)
            current_month = datetime.now().month
            if current_month in [6, 7, 8, 9]:  # Monsoon season
                risk_score += 0.1
            
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error in rule-based assessment: {str(e)}")
            return 0.5
    
    def _rule_based_outbreak_assessment(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Rule-based outbreak assessment fallback
        """
        risk_score = self._rule_based_risk_assessment(features)
        
        predicted_diseases = []
        disease_probabilities = {}
        
        if risk_score > 0.7:
            predicted_diseases = ['diarrhea', 'gastroenteritis']
            disease_probabilities = {'diarrhea': 0.6, 'gastroenteritis': 0.4}
        elif risk_score > 0.5:
            predicted_diseases = ['diarrhea']
            disease_probabilities = {'diarrhea': 0.4}
        
        return {
            'outbreak_probability': risk_score,
            'predicted_diseases': predicted_diseases,
            'disease_probabilities': disease_probabilities
        }
    
    def get_risk_factors(self, features: np.ndarray, risk_score: float) -> List[str]:
        """
        Identify primary risk factors contributing to the risk score
        """
        risk_factors = []
        
        try:
            # Water quality factors
            ph_level = features[0] if len(features) > 0 else 7.0
            turbidity = features[1] if len(features) > 1 else 0.0
            chlorine = features[2] if len(features) > 2 else 0.0
            symptom_count = features[5] if len(features) > 5 else 0
            sanitation_score = features[6] if len(features) > 6 else 5.0
            
            if ph_level < 6.5:
                risk_factors.append("Water pH too acidic")
            elif ph_level > 8.5:
                risk_factors.append("Water pH too alkaline")
            
            if turbidity > 5.0:
                risk_factors.append("High water turbidity")
            elif turbidity > 2.0:
                risk_factors.append("Elevated water turbidity")
            
            if chlorine < 0.2:
                risk_factors.append("Insufficient water chlorination")
            
            if symptom_count > 5:
                risk_factors.append("High number of reported symptoms in area")
            elif symptom_count > 2:
                risk_factors.append("Multiple symptom reports in area")
            
            if sanitation_score < 3:
                risk_factors.append("Poor sanitation infrastructure")
            elif sanitation_score < 5:
                risk_factors.append("Inadequate sanitation facilities")
            
            # Seasonal factors
            current_month = datetime.now().month
            if current_month in [6, 7, 8, 9]:
                risk_factors.append("Monsoon season - increased contamination risk")
            
            # If no specific factors identified but risk is high
            if not risk_factors and risk_score > 0.6:
                risk_factors.append("Multiple contributing factors detected")
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {str(e)}")
            risk_factors.append("Unable to determine specific risk factors")
        
        return risk_factors
    
    def load_model(self) -> bool:
        """
        Load pre-trained model from disk
        """
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.risk_model = model_data.get('risk_model')
                self.outbreak_model = model_data.get('outbreak_model')
                logger.info("Pre-trained model loaded successfully")
                
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Feature scaler loaded successfully")
                
            return True
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {str(e)}")
            logger.info("Using rule-based fallback methods")
            return False
    
    def save_model(self) -> bool:
        """
        Save trained model to disk
        """
        try:
            # Ensure model directory exists
            os.makedirs(settings.MODEL_PATH, exist_ok=True)
            
            # Save models
            model_data = {
                'risk_model': self.risk_model,
                'outbreak_model': self.outbreak_model,
                'feature_columns': self.feature_columns,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'version': '1.0'
            }
            
            joblib.dump(model_data, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            logger.info("Model saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def train_model(self, training_data: List[Dict[str, Any]], labels: List[float]) -> Dict[str, Any]:
        """
        Train the prediction model with new data
        """
        try:
            # Prepare features
            features_list = []
            for data_point in training_data:
                features = self.prepare_features(data_point)
                features_list.append(features)
            
            X = np.array(features_list)
            y_risk = np.array(labels)  # Risk scores
            
            # Create outbreak labels (binary classification)
            y_outbreak = (y_risk > 0.5).astype(int)
            
            # Split data
            X_train, X_test, y_risk_train, y_risk_test, y_outbreak_train, y_outbreak_test = train_test_split(
                X, y_risk, y_outbreak, test_size=0.2, random_state=42
            )
            
            # Fit scaler
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train risk prediction model (regression)
            self.risk_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.risk_model.fit(X_train_scaled, y_risk_train)
            
            # Train outbreak classification model
            self.outbreak_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.outbreak_model.fit(X_train_scaled, y_outbreak_train)
            
            # Evaluate models
            risk_score = self.risk_model.score(X_test_scaled, y_risk_test)
            outbreak_score = self.outbreak_model.score(X_test_scaled, y_outbreak_test)
            
            # Get feature importance
            feature_importance = dict(zip(
                self.feature_columns,
                self.risk_model.feature_importances_
            ))
            
            training_results = {
                'risk_model_r2_score': risk_score,
                'outbreak_model_accuracy': outbreak_score,
                'feature_importance': feature_importance,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'trained_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Save trained model
            self.save_model()
            
            logger.info(f"Model training completed. Risk R2: {risk_score:.3f}, Outbreak Accuracy: {outbreak_score:.3f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise


# Model instance (singleton)
disease_prediction_model = WaterborneDiseasePredictionModel()