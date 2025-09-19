import pytest
import firebase_admin
from firebase_admin import credentials, firestore
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional, AsyncGenerator
import logging
import os
import asyncio
import httpx
from unittest.mock import Mock, patch

from app.config import settings
from app.main import app
from app.dependencies import get_current_user

logger = logging.getLogger(__name__)

# Mock user data for tests
MOCK_USER = {
    "uid": "test_user_123",
    "email": "test@example.com",
    "role": "health_worker",
    "region": "Northeast India",
    "state": "Assam",
    "district": "Kamrup",
    "verified": True
}

# Override the get_current_user dependency for tests
async def mock_get_current_user():
    """Mock authentication for tests"""
    return MOCK_USER

# Override the dependency
app.dependency_overrides[get_current_user] = mock_get_current_user

@pytest.fixture(scope="session", autouse=True)
def initialize_firebase():
    """Initialize Firebase for tests with mock credentials"""
    try:
        # Clear any existing apps
        if firebase_admin._apps:
            for existing_app in firebase_admin._apps.values():
                firebase_admin.delete_app(existing_app)
        
        # Initialize with mock credentials for testing
        cred = Mock()
        firebase_admin.initialize_app(cred, name="test_app")
        logger.info("Firebase initialized for testing")
    except Exception as e:
        logger.warning(f"Could not initialize Firebase for tests: {e}")

@pytest.fixture(scope="session")
def firestore_service():
    """
    Fixture to provide mocked Firestore service for tests
    """
    from app.db.database import FirestoreService
    
    # Create a mock Firestore service
    service = FirestoreService()
    
    # Mock the database client
    mock_db = Mock()
    mock_collection = Mock()
    mock_doc_ref = Mock()
    
    # Set up mock chain
    mock_db.collection.return_value = mock_collection
    mock_collection.document.return_value = mock_doc_ref
    mock_collection.add.return_value = (None, mock_doc_ref)
    mock_doc_ref.id = "mock_document_id"
    mock_doc_ref.set.return_value = None
    mock_doc_ref.get.return_value = Mock(exists=True, to_dict=lambda: {"test": "data"}, id="test_id")
    
    # Mock query operations
    mock_stream = Mock()
    mock_stream.__iter__ = lambda x: iter([Mock(to_dict=lambda: {"test": "data"}, id="test_id")])
    mock_collection.stream.return_value = mock_stream
    mock_collection.where.return_value = mock_collection
    mock_collection.order_by.return_value = mock_collection
    mock_collection.limit.return_value = mock_collection
    
    service.db = mock_db
    return service

@pytest.fixture(scope="session")
def event_loop():
    """Creates an event loop for async tests"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Fixture for creating a test client for the FastAPI app"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for all tests"""
    with patch('app.core.cache.redis_client') as mock_redis:
        # Mock Redis operations
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = True
        mock_redis.ping.return_value = True
        
        with patch('app.db.database.firestore_service') as mock_firestore:
            # Mock Firestore operations
            mock_firestore.create_document.return_value = "mock_doc_id"
            mock_firestore.get_document.return_value = {"id": "test", "data": "test"}
            mock_firestore.update_document.return_value = True
            mock_firestore.query_collection.return_value = []
            
            yield {
                'redis': mock_redis,
                'firestore': mock_firestore
            }

@pytest.fixture
def mock_ml_predictor():
    """Mock ML predictor for tests"""
    with patch('app.ml.predictor.RiskPredictor') as mock_predictor:
        # Mock prediction result
        mock_prediction = Mock()
        mock_prediction.risk_score = 0.75
        mock_prediction.risk_level = "HIGH"
        mock_prediction.confidence = 0.85
        mock_prediction.predicted_diseases = ["diarrhea", "gastroenteritis"]
        mock_prediction.primary_risk_factors = ["High turbidity", "Multiple symptoms"]
        mock_prediction.model_version = "v1.0"
        
        mock_predictor.return_value.predict_risk.return_value = mock_prediction
        mock_predictor.return_value.batch_risk_assessment.return_value = [mock_prediction, mock_prediction]
        
        yield mock_predictor

@pytest.fixture
def sample_water_quality_data():
    """Sample water quality data for tests"""
    return {
        "sensor_id": "TEST_SENSOR_001",
        "location": {"latitude": 26.1445, "longitude": 91.7362},
        "water_source": "tube_well",
        "ph_level": 7.2,
        "turbidity": 1.5,
        "residual_chlorine": 0.2,
        "temperature": 25.5,
        "collection_method": "manual",
        "collected_by": "test_user",
        "notes": "Test data"
    }

@pytest.fixture
def sample_health_data():
    """Sample health data for tests"""
    return {
        "patient_name": "Test Patient",
        "age": 35,
        "gender": "female",
        "location": {"latitude": 26.1445, "longitude": 91.7362},
        "symptoms": ["diarrhea", "vomiting", "fever"],
        "symptom_severity": "moderate",
        "symptom_onset_date": "2024-01-15T08:00:00Z",
        "fever_temperature": 38.5,
        "water_source_used": "hand_pump",
        "recent_travel": False,
        "reported_by": "test_health_worker"
    }

@pytest.fixture
def sample_alert_data():
    """Sample alert data for tests"""
    return {
        "alert_type": "water_contamination",
        "severity": "high",
        "title": "Test Water Contamination Alert",
        "message": "Test message for water contamination alert",
        "location": {"latitude": 26.1445, "longitude": 91.7362},
        "affected_radius_km": 2.0,
        "target_audience": ["user", "health_worker"],
        "languages": ["en"],
        "action_required": True,
        "contact_info": "Contact test center: +91-1234567890"
    }

# Clean up after all tests
@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """Cleanup after all tests"""
    yield
    # Clean up any test data, close connections, etc.
    try:
        if firebase_admin._apps:
            for app_instance in firebase_admin._apps.values():
                firebase_admin.delete_app(app_instance)
    except:
        pass