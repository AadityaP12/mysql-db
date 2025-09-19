from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DiseaseType(str, Enum):
    DIARRHEA = "diarrhea"
    CHOLERA = "cholera"
    TYPHOID = "typhoid"
    HEPATITIS_A = "hepatitis_a"
    DYSENTERY = "dysentery"
    GASTROENTERITIS = "gastroenteritis"
    GENERAL_WATERBORNE = "general_waterborne"


class WaterSourceType(str, Enum):
    TUBE_WELL = "tube_well"
    HAND_PUMP = "hand_pump"
    DUG_WELL = "dug_well"
    SPRING = "spring"
    SURFACE_WATER = "surface_water"
    PIPED_WATER = "piped_water"
    RAINWATER = "rainwater"
    OTHER = "other"


class SanitationType(str, Enum):
    IMPROVED = "improved"
    UNIMPROVED = "unimproved"
    OPEN_DEFECATION = "open_defecation"


class PredictionInput(BaseModel):
    location: Dict[str, float] = Field(..., description="Latitude and longitude")
    population_density: Optional[int] = Field(None, ge=0, description="People per sq km")
    water_quality_data: Dict[str, Union[float, str]] = Field(
        ..., 
        description="Water quality metrics including pH, turbidity, etc."
    )
    health_symptoms_count: Optional[Dict[str, int]] = Field(
        default_factory=dict, 
        description="Count of reported symptoms in area"
    )
    seasonal_factors: Optional[Dict[str, Any]] = Field(
        None, 
        description="Current season and climate factors"
    )
    historical_outbreak_data: Optional[List[Dict]] = Field(
        None, 
        description="Previous outbreak data for the area"
    )
    sanitation_score: Optional[float] = Field(None, ge=0, le=10, description="Sanitation quality score")
    income_level: Optional[float] = Field(None, ge=0, description="Average household income")
    age_demographics: Optional[Dict[str, float]] = Field(
        None, 
        description="Age group distribution"
    )
    population_age_median: Optional[int] = Field(None, ge=0, le=120, description="Median age of population")
    water_source: Optional[WaterSourceType] = Field(None, description="Primary water source type")
    time_period: str = Field(default="current", pattern="^(current|1_week|1_month)$")
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v):
        if 'latitude' not in v or 'longitude' not in v:
            raise ValueError('Location must contain latitude and longitude')
        
        # Validate coordinates for Northeast India region
        lat, lon = v['latitude'], v['longitude']
        if not (21.0 <= lat <= 30.0 and 87.0 <= lon <= 98.0):
            raise ValueError('Coordinates must be within Northeast India region')
        return v
    
    @field_validator('water_quality_data')
    @classmethod
    def validate_water_quality(cls, v):
        # Ensure at least some basic water quality parameters
        if not any(key in v for key in ['ph_level', 'ph', 'turbidity']):
            raise ValueError('Water quality data must contain at least pH or turbidity')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "population_density": 250,
                "water_quality_data": {
                    "ph_level": 7.2,
                    "turbidity": 1.5,
                    "residual_chlorine": 0.2,
                    "hardness": 200.0,
                    "chloramines": 3.5,
                    "source_type": "piped"
                },
                "health_symptoms_count": {
                    "diarrhea": 5,
                    "vomiting": 3,
                    "fever": 7
                },
                "sanitation_score": 6.5,
                "income_level": 45000,
                "population_age_median": 32,
                "time_period": "current"
            }
        }


class RiskPrediction(BaseModel):
    location: Dict[str, float]
    risk_score: float = Field(..., ge=0, le=1, description="Risk score between 0 and 1")
    risk_level: RiskLevel
    risk_color: str = Field(..., description="Color code for risk visualization")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence in prediction")
    predicted_diseases: List[DiseaseType] = Field(..., description="Most likely diseases")
    primary_risk_factors: List[str] = Field(..., description="Main contributing risk factors")
    population_at_risk: Optional[int] = Field(None, ge=0, description="Estimated people at risk")
    model_version: str = Field(default="v1.0", description="Version of ML model used")
    prediction_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Additional fields for enhanced predictions
    risk_percent: Optional[float] = Field(None, ge=0, le=100, description="Risk as percentage")
    recommendations: Optional[List[str]] = Field(None, description="Recommended actions")
    affected_radius_km: Optional[float] = Field(None, ge=0, description="Estimated affected area radius")
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "risk_score": 0.75,
                "risk_level": "high",
                "risk_color": "RED",
                "confidence": 0.85,
                "predicted_diseases": ["diarrhea", "gastroenteritis"],
                "primary_risk_factors": [
                    "High turbidity levels",
                    "Increased symptom reports",
                    "Poor sanitation score"
                ],
                "population_at_risk": 150,
                "model_version": "rf_ensemble_v1.0",
                "risk_percent": 75.0,
                "affected_radius_km": 3.5
            }
        }


class OutbreakPrediction(BaseModel):
    region_id: str
    outbreak_probability: float = Field(..., ge=0, le=1, description="Probability of outbreak occurring")
    expected_cases: int = Field(..., ge=0, description="Estimated number of cases")
    peak_time_estimate: Optional[datetime] = Field(None, description="When outbreak might peak")
    disease_type: DiseaseType
    affected_radius_km: float = Field(..., ge=0, description="Expected affected area radius")
    preventive_measures: List[str] = Field(..., description="Recommended preventive actions")
    resource_requirements: Dict[str, int] = Field(..., description="Estimated resource needs")
    prediction_accuracy: float = Field(..., ge=0, le=1, description="Expected prediction accuracy")
    
    # Enhanced outbreak prediction fields
    transmission_rate: Optional[float] = Field(None, ge=0, description="Estimated transmission rate")
    severity_level: Optional[str] = Field(None, description="Expected outbreak severity")
    duration_estimate_days: Optional[int] = Field(None, ge=0, description="Expected outbreak duration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "region_id": "ASSAM_KAMRUP_001",
                "outbreak_probability": 0.65,
                "expected_cases": 25,
                "peak_time_estimate": "2024-02-01T00:00:00Z",
                "disease_type": "diarrhea",
                "affected_radius_km": 5.0,
                "preventive_measures": [
                    "Boil water before consumption",
                    "Increase chlorination",
                    "Health education campaigns"
                ],
                "resource_requirements": {
                    "ors_packets": 100,
                    "water_purification_tablets": 500,
                    "health_workers": 3
                },
                "prediction_accuracy": 0.78,
                "severity_level": "moderate",
                "duration_estimate_days": 14
            }
        }


class WaterQualityPredictionInput(BaseModel):
    """Specific input schema for water quality risk prediction"""
    location: Dict[str, float] = Field(..., description="Latitude and longitude")
    ph_level: Optional[float] = Field(7.0, ge=0, le=14, description="pH level")
    turbidity: Optional[float] = Field(2.0, ge=0, description="Turbidity in NTU")
    hardness: Optional[float] = Field(200.0, ge=0, description="Water hardness")
    solids: Optional[float] = Field(20000.0, ge=0, description="Total dissolved solids")
    chloramines: Optional[float] = Field(7.0, ge=0, description="Chloramine content")
    sulfate: Optional[float] = Field(300.0, ge=0, description="Sulfate content")
    conductivity: Optional[float] = Field(400.0, ge=0, description="Electrical conductivity")
    organic_carbon: Optional[float] = Field(14.0, ge=0, description="Total organic carbon")
    trihalomethanes: Optional[float] = Field(80.0, ge=0, description="Trihalomethane content")
    source_type: Optional[str] = Field("surface", description="Water source type")
    season: Optional[str] = Field("summer", description="Current season")
    month: Optional[int] = Field(6, ge=1, le=12, description="Current month")
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "ph_level": 7.2,
                "turbidity": 1.5,
                "hardness": 180.0,
                "solids": 18500.0,
                "chloramines": 6.8,
                "sulfate": 280.0,
                "conductivity": 385.0,
                "organic_carbon": 12.5,
                "trihalomethanes": 75.0,
                "source_type": "surface",
                "season": "monsoon",
                "month": 7
            }
        }


class HealthRiskPredictionInput(BaseModel):
    """Specific input schema for health risk prediction"""
    location: Dict[str, float] = Field(..., description="Latitude and longitude")
    age: Optional[int] = Field(30, ge=0, le=120, description="Age of individual or median age")
    sex: Optional[str] = Field("male", pattern="^(male|female|mixed)$", description="Sex/gender")
    income: Optional[float] = Field(50000.0, ge=0, description="Household income")
    sanitation: Optional[str] = Field("improved", description="Sanitation type")
    water_source: Optional[str] = Field("piped", description="Primary water source")
    water_quality_data: Dict[str, float] = Field(..., description="Water quality parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "age": 35,
                "sex": "female",
                "income": 45000.0,
                "sanitation": "improved",
                "water_source": "piped",
                "water_quality_data": {
                    "ph_level": 7.2,
                    "turbidity": 1.8,
                    "hardness": 190.0,
                    "chloramines": 6.5
                }
            }
        }


class ModelTrainingData(BaseModel):
    features: List[Dict[str, Any]]
    labels: List[float]
    data_source: str
    training_period: Dict[str, datetime]
    model_type: str = Field(default="random_forest")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    {
                        "ph_level": 7.2,
                        "turbidity": 1.5,
                        "symptom_count": 5,
                        "population_density": 250
                    }
                ],
                "labels": [0.75],
                "data_source": "northeastern_regions",
                "training_period": {
                    "start_date": "2023-01-01T00:00:00Z",
                    "end_date": "2024-01-01T00:00:00Z"
                },
                "model_type": "random_forest"
            }
        }


class ModelPerformance(BaseModel):
    model_id: str
    model_version: str
    model_type: str = Field(..., description="Type of ML model (RF, XGB, etc.)")
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    r2_score: Optional[float] = Field(None, ge=-1, le=1, description="R-squared score for regression")
    mae: Optional[float] = Field(None, ge=0, description="Mean Absolute Error")
    rmse: Optional[float] = Field(None, ge=0, description="Root Mean Square Error")
    auc_roc: Optional[float] = Field(None, ge=0, le=1)
    confusion_matrix: Optional[List[List[int]]] = None
    training_date: datetime
    validation_samples: int
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "water_quality_risk_model",
                "model_version": "rf_v1.2",
                "model_type": "RandomForest",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
                "r2_score": 0.78,
                "mae": 0.12,
                "rmse": 0.18,
                "auc_roc": 0.91,
                "training_date": "2024-01-15T10:00:00Z",
                "validation_samples": 500,
                "feature_importance": {
                    "ph_level": 0.25,
                    "turbidity": 0.22,
                    "chloramines": 0.18,
                    "hardness": 0.15,
                    "conductivity": 0.12,
                    "age": 0.08
                }
            }
        }


class PredictionHistory(BaseModel):
    prediction_id: str
    location: Dict[str, float]
    prediction_date: datetime
    prediction_type: str = Field(..., pattern="^(risk|outbreak|water_quality|health_risk)$")
    risk_score: Optional[float] = Field(None, ge=0, le=1)
    risk_level: Optional[RiskLevel] = None
    actual_outcome: Optional[bool] = Field(None, description="Whether prediction was correct")
    accuracy_score: Optional[float] = Field(None, ge=0, le=1)
    model_version: str
    validated: bool = Field(default=False)
    validated_by: Optional[str] = None
    validated_at: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "pred_123456789",
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "prediction_date": "2024-01-15T10:00:00Z",
                "prediction_type": "risk",
                "risk_score": 0.75,
                "risk_level": "high",
                "actual_outcome": True,
                "accuracy_score": 0.85,
                "model_version": "rf_ensemble_v1.0",
                "validated": True,
                "validated_by": "health_worker_001"
            }
        }


class BatchPredictionRequest(BaseModel):
    locations: List[Dict[str, float]] = Field(..., min_length=1, max_length=50)
    prediction_type: str = Field(default="risk_assessment", pattern="^(risk_assessment|outbreak_prediction|water_quality|health_risk)$")
    time_horizon: str = Field(default="1_week", pattern="^(current|1_week|1_month)$")
    include_recommendations: bool = Field(default=True)
    water_quality_defaults: Optional[Dict[str, float]] = Field(None, description="Default water quality values to use")
    population_defaults: Optional[Dict[str, float]] = Field(None, description="Default population parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "locations": [
                    {"latitude": 26.1445, "longitude": 91.7362},
                    {"latitude": 26.2045, "longitude": 91.8162}
                ],
                "prediction_type": "risk_assessment",
                "time_horizon": "1_week",
                "include_recommendations": True,
                "water_quality_defaults": {
                    "ph_level": 7.0,
                    "turbidity": 2.0
                }
            }
        }


class ModelRetrainRequest(BaseModel):
    model_type: str = Field(..., pattern="^(water_quality|health_risk|combined)$")
    data_source_period: Dict[str, datetime]
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    validation_method: str = Field(default="cross_validation", pattern="^(cross_validation|hold_out|bootstrap)$")
    requested_by: str
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    include_feature_selection: bool = Field(default=True, description="Whether to perform feature selection")
    target_metric: str = Field(default="f1_score", description="Primary metric to optimize")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "water_quality",
                "data_source_period": {
                    "start_date": "2023-06-01T00:00:00Z",
                    "end_date": "2024-01-01T00:00:00Z"
                },
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 12,
                    "min_samples_split": 2
                },
                "validation_method": "cross_validation",
                "requested_by": "admin_user",
                "priority": "normal",
                "target_metric": "r2_score"
            }
        }


class ModelValidationRequest(BaseModel):
    prediction_id: str
    actual_outcome: bool
    outcome_confidence: float = Field(1.0, ge=0, le=1, description="Confidence in the actual outcome")
    validation_notes: Optional[str] = Field(None, max_length=500)
    validation_source: str = Field(..., description="Source of validation data")
    validation_date: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "pred_123456789",
                "actual_outcome": True,
                "outcome_confidence": 0.95,
                "validation_notes": "Confirmed outbreak occurred as predicted",
                "validation_source": "field_verification",
                "validation_date": "2024-01-20T10:00:00Z"
            }
        }


class CombinedRiskAssessment(BaseModel):
    """Combined assessment using both water quality and health risk models"""
    location: Dict[str, float]
    water_quality_risk: Optional[RiskPrediction] = None
    health_risk: Optional[RiskPrediction] = None
    combined_risk_score: float = Field(..., ge=0, le=1)
    combined_risk_level: RiskLevel
    confidence: float = Field(..., ge=0, le=1)
    primary_risk_factors: List[str]
    recommended_actions: List[str]
    resource_requirements: Optional[Dict[str, int]] = None
    monitoring_recommendations: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "combined_risk_score": 0.72,
                "combined_risk_level": "high",
                "confidence": 0.84,
                "primary_risk_factors": [
                    "Poor water quality indicators",
                    "High population density",
                    "Inadequate sanitation"
                ],
                "recommended_actions": [
                    "Implement water treatment",
                    "Increase health monitoring",
                    "Deploy rapid response team"
                ],
                "monitoring_recommendations": [
                    "Daily water quality testing",
                    "Weekly health surveys",
                    "Real-time symptom tracking"
                ]
            }
        }


class FeatureImportanceAnalysis(BaseModel):
    """Analysis of feature importance from ML models"""
    model_version: str
    feature_rankings: Dict[str, float] = Field(..., description="Feature names mapped to importance scores")
    top_features: List[str] = Field(..., description="Top 10 most important features")
    feature_categories: Dict[str, List[str]] = Field(..., description="Features grouped by category")
    analysis_date: datetime = Field(default_factory=datetime.now)
    interpretation: List[str] = Field(..., description="Human-readable interpretation of feature importance")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_version": "rf_water_v1.0",
                "feature_rankings": {
                    "ph_level": 0.25,
                    "turbidity": 0.22,
                    "chloramines": 0.18,
                    "age": 0.12,
                    "sanitation": 0.10
                },
                "top_features": ["ph_level", "turbidity", "chloramines"],
                "feature_categories": {
                    "water_quality": ["ph_level", "turbidity", "chloramines"],
                    "demographic": ["age", "income"],
                    "infrastructure": ["sanitation", "water_source"]
                },
                "interpretation": [
                    "Water pH is the strongest predictor of health risk",
                    "Turbidity levels significantly impact disease probability",
                    "Chemical disinfectants play an important protective role"
                ]
            }
        }