from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
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


class PredictionInput(BaseModel):
    location: Dict[str, float] = Field(..., description="Latitude and longitude")
    population_density: Optional[int] = Field(None, ge=0)
    water_quality_data: Dict[str, float] = Field(..., description="Recent water quality metrics")
    health_symptoms_count: Dict[str, int] = Field(default_factory=dict, description="Count of symptoms in area")
    seasonal_factors: Optional[Dict[str, Any]] = None
    historical_outbreak_data: Optional[List[Dict]] = None
    sanitation_score: Optional[float] = Field(None, ge=0, le=10)
    time_period: str = Field(default="current", pattern="^(current|1_week|1_month)$")
    
    @validator('location')
    def validate_location(cls, v):
        if 'latitude' not in v or 'longitude' not in v:
            raise ValueError('Location must contain latitude and longitude')
        return v
    
    @validator('water_quality_data')
    def validate_water_quality(cls, v):
        required_fields = ['ph_level', 'turbidity']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Water quality data must contain {field}')
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
                    "bacterial_contamination": 0
                },
                "health_symptoms_count": {
                    "diarrhea": 5,
                    "vomiting": 3,
                    "fever": 7
                },
                "sanitation_score": 6.5,
                "time_period": "current"
            }
        }


class RiskPrediction(BaseModel):
    location: Dict[str, float]
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    risk_color: str
    confidence: float = Field(..., ge=0, le=1)
    predicted_diseases: List[DiseaseType]
    primary_risk_factors: List[str]
    population_at_risk: Optional[int] = None
    prediction_timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(default="v1.0")
    
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
                "model_version": "v1.0"
            }
        }


class OutbreakPrediction(BaseModel):
    region_id: str
    outbreak_probability: float = Field(..., ge=0, le=1)
    expected_cases: int = Field(..., ge=0)
    peak_time_estimate: Optional[datetime] = None
    disease_type: DiseaseType
    affected_radius_km: float = Field(..., ge=0)
    preventive_measures: List[str]
    resource_requirements: Dict[str, int]
    prediction_accuracy: float = Field(..., ge=0, le=1)
    
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
                "prediction_accuracy": 0.78
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
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    auc_roc: Optional[float] = Field(None, ge=0, le=1)
    confusion_matrix: Optional[List[List[int]]] = None
    training_date: datetime
    validation_samples: int
    feature_importance: Optional[Dict[str, float]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "waterborne_disease_model",
                "model_version": "v1.2",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
                "auc_roc": 0.91,
                "training_date": "2024-01-15T10:00:00Z",
                "validation_samples": 500,
                "feature_importance": {
                    "water_quality": 0.35,
                    "symptom_reports": 0.25,
                    "seasonal_factors": 0.20,
                    "sanitation_score": 0.20
                }
            }
        }


class PredictionHistory(BaseModel):
    prediction_id: str
    location: Dict[str, float]
    prediction_date: datetime
    risk_score: float
    risk_level: RiskLevel
    actual_outcome: Optional[bool] = None  # Whether outbreak actually occurred
    accuracy_score: Optional[float] = None
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "pred_123456789",
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "prediction_date": "2024-01-15T10:00:00Z",
                "risk_score": 0.75,
                "risk_level": "high",
                "actual_outcome": True,
                "accuracy_score": 0.85,
                "model_version": "v1.0"
            }
        }


class BatchPredictionRequest(BaseModel):
    locations: List[Dict[str, float]] = Field(..., min_items=1, max_items=50)
    prediction_type: str = Field(default="risk_assessment", pattern="^(risk_assessment|outbreak_prediction)$")
    time_horizon: str = Field(default="1_week", pattern="^(current|1_week|1_month)$")
    include_recommendations: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "locations": [
                    {"latitude": 26.1445, "longitude": 91.7362},
                    {"latitude": 26.2045, "longitude": 91.8162}
                ],
                "prediction_type": "risk_assessment",
                "time_horizon": "1_week",
                "include_recommendations": True
            }
        }


class ModelRetrainRequest(BaseModel):
    model_type: str = Field(..., pattern="^(risk_prediction|outbreak_prediction)$")
    data_source_period: Dict[str, datetime]
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_method: str = Field(default="cross_validation")
    requested_by: str
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "risk_prediction",
                "data_source_period": {
                    "start_date": "2023-06-01T00:00:00Z",
                    "end_date": "2024-01-01T00:00:00Z"
                },
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10
                },
                "requested_by": "admin_user",
                "priority": "normal"
            }
        }