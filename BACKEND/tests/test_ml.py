import pytest
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_ml_risk_prediction(client):
    """Test ML risk prediction endpoint"""
    headers = {"Authorization": "Bearer faketoken"}
    payload = {
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
    
    # Mock the ML predictor to return a predictable result
    with patch('app.ml.predictor.RiskPredictor.predict_risk') as mock_predict:
        mock_prediction = Mock()
        mock_prediction.risk_score = 0.75
        mock_prediction.risk_level = "HIGH"
        mock_prediction.confidence = 0.85
        mock_prediction.predicted_diseases = ["diarrhea", "gastroenteritis"]
        mock_prediction.primary_risk_factors = ["High turbidity", "Multiple symptoms"]
        mock_prediction.model_version = "v1.0"
        mock_predict.return_value = mock_prediction
        
        res = await client.post("/api/v1/ml/predict/risk", json=payload, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] == True
        assert "risk_score" in str(data)

@pytest.mark.asyncio
async def test_ml_batch_prediction(client):
    """Test batch risk prediction endpoint"""
    headers = {"Authorization": "Bearer faketoken"}
    payload = {
        "locations": [
            {"latitude": 26.1445, "longitude": 91.7362},
            {"latitude": 26.2045, "longitude": 91.8162}
        ],
        "prediction_type": "risk_assessment",
        "time_horizon": "1_week",
        "include_recommendations": True
    }
    
    with patch('app.ml.predictor.RiskPredictor.batch_risk_assessment') as mock_batch:
        mock_predictions = []
        for i in range(2):
            mock_pred = Mock()
            mock_pred.risk_score = 0.6 + i * 0.1
            mock_pred.risk_level = "MEDIUM"
            mock_pred.confidence = 0.8
            mock_predictions.append(mock_pred)
        
        mock_batch.return_value = mock_predictions
        
        res = await client.post("/api/v1/ml/predict/batch", json=payload, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] == True
        assert "predictions" in data.get("data", {})

@pytest.mark.asyncio
async def test_ml_model_performance(client):
    """Test model performance endpoint"""
    headers = {"Authorization": "Bearer faketoken"}
    
    res = await client.get("/api/v1/ml/model/performance", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] == True
    assert "model_info" in data.get("data", {})

@pytest.mark.asyncio
async def test_ml_prediction_validation(client):
    """Test ML prediction with invalid data"""
    headers = {"Authorization": "Bearer faketoken"}
    # Invalid payload - missing required fields
    invalid_payload = {
        "location": {"latitude": 26.1445, "longitude": 91.7362},
        "water_quality_data": {
            "ph_level": 7.2
            # Missing required 'turbidity' field
        }
    }
    
    res = await client.post("/api/v1/ml/predict/risk", json=invalid_payload, headers=headers)
    assert res.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_ml_risk_map(client):
    """Test regional risk map generation"""
    headers = {"Authorization": "Bearer faketoken"}
    
    # Test with valid coordinate bounds for Northeast India
    params = {
        "lat_min": 26.0,
        "lat_max": 27.0,
        "lon_min": 91.0,
        "lon_max": 92.0
    }
    
    with patch('app.ml.predictor.RiskPredictor.get_regional_risk_map') as mock_risk_map:
        mock_risk_map.return_value = {
            "region_bounds": params,
            "risk_grid": [[0.3, 0.5], [0.7, 0.4]],
            "high_risk_areas": 1,
            "generated_at": "2024-01-15T10:00:00Z"
        }
        
        res = await client.get("/api/v1/ml/risk-map", params=params, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] == True