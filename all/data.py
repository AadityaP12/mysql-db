from pydantic import BaseModel, Field, validator
from pydantic import field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class WaterSource(str, Enum):
    TUBE_WELL = "tube_well"
    HAND_PUMP = "hand_pump"
    DUG_WELL = "dug_well"
    SPRING = "spring"
    SURFACE_WATER = "surface_water"
    PIPED_WATER = "piped_water"
    RAINWATER = "rainwater"
    OTHER = "other"


class SanitationType(str, Enum):
    FLUSH_TOILET = "flush_toilet"
    PIT_LATRINE = "pit_latrine"
    COMPOSTING_TOILET = "composting_toilet"
    NO_FACILITY = "no_facility"
    OPEN_DEFECATION = "open_defecation"
    OTHER = "other"


class SymptomSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class WaterQualityData(BaseModel):
    sensor_id: Optional[str] = None
    location: Dict[str, float] = Field(..., description="Latitude and longitude")
    water_source: WaterSource
    ph_level: float = Field(..., ge=0, le=14)
    turbidity: float = Field(..., ge=0)
    residual_chlorine: Optional[float] = Field(None, ge=0)
    temperature: Optional[float] = None
    dissolved_oxygen: Optional[float] = None
    bacterial_contamination: Optional[bool] = None
    ecoli_count: Optional[int] = Field(None, ge=0)
    total_coliform: Optional[int] = Field(None, ge=0)
    collection_method: str = Field(default="manual")  # "manual" or "iot_sensor"
    collected_by: str
    collection_time: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None
    
    @validator('location')
    def validate_location(cls, v):
        if 'latitude' not in v or 'longitude' not in v:
            raise ValueError('Location must contain latitude and longitude')
        
        # Validate coordinates for Northeast India region
        lat, lon = v['latitude'], v['longitude']
        if not (21.0 <= lat <= 30.0 and 87.0 <= lon <= 98.0):
            raise ValueError('Coordinates must be within Northeast India region')
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "sensor_id": "IOT_001",
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "water_source": "tube_well",
                "ph_level": 7.2,
                "turbidity": 1.5,
                "residual_chlorine": 0.2,
                "temperature": 25.5,
                "bacterial_contamination": False,
                "ecoli_count": 0,
                "collection_method": "iot_sensor",
                "collected_by": "IOT_SENSOR_001",
                "notes": "Water quality within acceptable limits"
            }
        }


class HouseholdData(BaseModel):
    house_id: str
    head_of_household: str
    total_members: int = Field(..., ge=1)
    location: Dict[str, float]
    water_source: WaterSource
    water_source_distance: Optional[float] = Field(None, description="Distance in meters")
    water_treatment_method: Optional[str] = None
    sanitation_type: SanitationType
    waste_disposal_method: Optional[str] = None
    electricity_available: bool = False
    cooking_fuel: Optional[str] = None
    household_income_bracket: Optional[str] = None
    education_level_hoh: Optional[str] = None  # Head of household
    registered_by: str
    registration_date: datetime = Field(default_factory=datetime.now)
    
    @field_validator('location')
    def validate_location(cls, v):
        if 'latitude' not in v or 'longitude' not in v:
            raise ValueError('Location must contain latitude and longitude')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "house_id": "HH_001_VILLAGE",
                "head_of_household": "Ram Kumar",
                "total_members": 5,
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "water_source": "hand_pump",
                "water_source_distance": 50,
                "sanitation_type": "pit_latrine",
                "electricity_available": True,
                "registered_by": "asha_worker_001"
            }
        }


class HealthData(BaseModel):
    patient_name: str = Field(..., min_length=2)
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    household_id: Optional[str] = None
    location: Dict[str, float]
    symptoms: List[str] = Field(..., min_items=1)
    symptom_severity: SymptomSeverity = SymptomSeverity.MILD
    symptom_onset_date: datetime
    fever_temperature: Optional[float] = None
    dehydration_level: Optional[str] = None
    hospitalization_required: bool = False
    diagnosis: Optional[str] = None
    treatment_given: Optional[str] = None
    water_source_used: WaterSource
    recent_travel: bool = False
    travel_locations: Optional[List[str]] = None
    contact_with_cases: bool = False
    reported_by: str
    report_time: datetime = Field(default_factory=datetime.now)
    follow_up_required: bool = False
    notes: Optional[str] = None
    
    @validator('fever_temperature')
    def validate_fever_temperature(cls, v):
        if v is not None and (v < 35.0 or v > 45.0):
            raise ValueError('Fever temperature must be between 35°C and 45°C')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_name": "Anita Devi",
                "age": 35,
                "gender": "female",
                "household_id": "HH_001_VILLAGE",
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "symptoms": ["diarrhea", "vomiting", "fever"],
                "symptom_severity": "moderate",
                "symptom_onset_date": "2024-01-15T08:00:00Z",
                "fever_temperature": 38.5,
                "water_source_used": "hand_pump",
                "recent_travel": False,
                "reported_by": "asha_worker_001",
                "notes": "Patient shows signs of dehydration"
            }
        }


class SensorData(BaseModel):
    sensor_id: str
    sensor_type: str = Field(..., pattern="^(water_quality|weather|environmental)$")
    location: Dict[str, float]
    readings: Dict[str, Any] = Field(..., description="Sensor readings as key-value pairs")
    timestamp: datetime = Field(default_factory=datetime.now)
    battery_level: Optional[float] = Field(None, ge=0, le=100)
    signal_strength: Optional[int] = None
    calibration_date: Optional[datetime] = None
    status: str = Field(default="active")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sensor_id": "WQ_SENSOR_001",
                "sensor_type": "water_quality",
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "readings": {
                    "ph": 7.2,
                    "turbidity": 1.5,
                    "temperature": 25.5,
                    "residual_chlorine": 0.2
                },
                "battery_level": 85.5,
                "signal_strength": -65,
                "status": "active"
            }
        }


class DataUploadResponse(BaseModel):
    success: bool
    message: str
    data_id: str
    processed_at: datetime
    validation_errors: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Data uploaded successfully",
                "data_id": "data_123456789",
                "processed_at": "2024-01-15T10:30:00Z",
                "validation_errors": []
            }
        }


class DataQuery(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    region: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    data_type: Optional[str] = Field(None, pattern="^(health|water_quality|sensor|household)$")
    limit: int = Field(default=50, le=1000)
    offset: int = Field(default=0, ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T23:59:59Z",
                "state": "Assam",
                "district": "Kamrup",
                "data_type": "health",
                "limit": 100
            }
        }


class BulkDataUpload(BaseModel):
    data_type: str = Field(..., pattern="^(health|water_quality|household|sensor)$")
    data_records: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100)
    uploaded_by: str
    upload_source: str = Field(default="mobile_app")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data_type": "health",
                "data_records": [
                    {
                        "patient_name": "John Doe",
                        "age": 30,
                        "symptoms": ["fever", "headache"]
                    }
                ],
                "uploaded_by": "health_worker_001",
                "upload_source": "mobile_app"
            }
        }