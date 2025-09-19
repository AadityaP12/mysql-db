from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.schemas.ml import (
    PredictionInput, RiskPrediction, OutbreakPrediction,
    BatchPredictionRequest, ModelTrainingData, ModelPerformance,
    PredictionHistory, ModelRetrainRequest
)
from app.dependencies import get_current_user, get_admin_user, get_health_worker
from app.ml.predictor import RiskPredictor
from app.ml.model import disease_prediction_model
from app.db.database import firestore_service, FirestoreCollections
from app.core.utils import create_response, generate_id

router = APIRouter()
risk_predictor = RiskPredictor()


@router.post("/predict/risk", response_model=RiskPrediction)
async def predict_risk(
    prediction_input: PredictionInput,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate risk prediction for waterborne disease outbreak
    """
    try:
        # Convert Pydantic model to dict for processing
        input_data = prediction_input.dict()
        
        # Generate risk prediction
        risk_prediction = await risk_predictor.predict_risk(input_data)
        
        # Log prediction request
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": current_user["uid"],
                "action": "risk_prediction_requested",
                "location": input_data["location"],
                "prediction_id": f"PRED_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        return risk_prediction
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk prediction failed: {str(e)}"
        )


@router.post("/predict/outbreak")
async def predict_outbreak(
    region_id: str,
    prediction_input: PredictionInput,
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Generate outbreak prediction for a specific region
    """
    try:
        # Convert input to dict
        input_data = prediction_input.dict()
        
        # Generate outbreak prediction
        outbreak_prediction = await risk_predictor.predict_outbreak(region_id, input_data)
        
        # Log prediction request
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": current_user["uid"],
                "action": "outbreak_prediction_requested",
                "region_id": region_id,
                "prediction_id": f"OUTBREAK_PRED_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        return create_response(
            success=True,
            message="Outbreak prediction generated successfully",
            data=outbreak_prediction.dict()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Outbreak prediction failed: {str(e)}"
        )


@router.post("/predict/batch")
async def batch_risk_prediction(
    batch_request: BatchPredictionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate risk predictions for multiple locations
    """
    try:
        if len(batch_request.locations) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 locations allowed per batch request"
            )
        
        # Generate predictions for all locations
        predictions = await risk_predictor.batch_risk_assessment(batch_request.locations)
        
        # Convert predictions to dict format
        prediction_data = [pred.dict() for pred in predictions]
        
        # Log batch request
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": current_user["uid"],
                "action": "batch_prediction_requested",
                "location_count": len(batch_request.locations),
                "prediction_type": batch_request.prediction_type,
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        return create_response(
            success=True,
            message=f"Batch prediction completed for {len(predictions)} locations",
            data={
                "predictions": prediction_data,
                "total_locations": len(batch_request.locations),
                "successful_predictions": len(predictions),
                "prediction_type": batch_request.prediction_type,
                "time_horizon": batch_request.time_horizon
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/risk-map")
async def get_regional_risk_map(
    lat_min: float = Query(..., ge=21.0, le=30.0),
    lat_max: float = Query(..., ge=21.0, le=30.0),
    lon_min: float = Query(..., ge=87.0, le=98.0),
    lon_max: float = Query(..., ge=98.0, le=98.0),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate risk map for a geographic region
    """
    try:
        # Validate bounds
        if lat_min >= lat_max or lon_min >= lon_max:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid coordinate bounds"
            )
        
        region_bounds = {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        }
        
        # Generate risk map
        risk_map_data = await risk_predictor.get_regional_risk_map(region_bounds)
        
        return create_response(
            success=True,
            message="Regional risk map generated successfully",
            data=risk_map_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk map generation failed: {str(e)}"
        )


@router.get("/predictions/history")
async def get_prediction_history(
    limit: int = Query(50, le=200),
    prediction_type: Optional[str] = Query(None, pattern="^(risk|outbreak)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get historical predictions with filters
    """
    try:
        filters = []
        
        # Apply date filters
        if start_date:
            filters.append(("prediction_timestamp", ">=", start_date))
        if end_date:
            filters.append(("prediction_timestamp", "<=", end_date))
        
        # Apply type filter
        if prediction_type:
            filters.append(("prediction_type", "==", prediction_type))
        
        # Apply location-based restrictions for non-admin users
        user_role = current_user.get("role")
        if user_role not in ["admin", "state_health_authority"]:
            # Add location filters based on user's region
            pass  # Implementation depends on how predictions store location data
        
        # Query predictions
        predictions = await firestore_service.query_collection(
            FirestoreCollections.ML_PREDICTIONS,
            filters=filters,
            order_by="prediction_timestamp",
            limit=limit
        )
        
        # Sort by timestamp descending
        predictions.sort(key=lambda x: x.get("prediction_timestamp"), reverse=True)
        
        return create_response(
            success=True,
            message="Prediction history retrieved successfully",
            data=predictions,
            meta={"count": len(predictions), "limit": limit}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction history: {str(e)}"
        )


@router.get("/model/performance")
async def get_model_performance(
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Get current model performance metrics
    """
    try:
        # Get model information
        model_info = {
            "model_version": getattr(disease_prediction_model, 'version', 'v1.0'),
            "model_type": "ensemble_waterborne_disease_predictor",
            "feature_count": len(disease_prediction_model.feature_columns),
            "training_features": disease_prediction_model.feature_columns,
            "last_updated": datetime.now(timezone.utc).isoformat(),  # Would be actual last update
        }
        
        # Get recent prediction accuracy if available
        recent_predictions = await firestore_service.query_collection(
            FirestoreCollections.ML_PREDICTIONS,
            filters=[
                ("prediction_timestamp", ">=", datetime.now(timezone.utc).replace(day=1))  # This month
            ],
            limit=1000
        )
        
        # Calculate accuracy metrics (simplified)
        accuracy_metrics = {
            "total_predictions": len(recent_predictions),
            "accuracy_estimate": 0.78,  # Would calculate from actual vs predicted
            "precision_estimate": 0.75,
            "recall_estimate": 0.82,
            "f1_score_estimate": 0.78
        }
        
        # Feature importance (mock data - would come from actual model)
        feature_importance = {
            "water_quality_ph": 0.25,
            "symptom_reports": 0.22,
            "turbidity_level": 0.18,
            "sanitation_score": 0.15,
            "seasonal_factors": 0.12,
            "population_density": 0.08
        }
        
        performance_data = {
            "model_info": model_info,
            "accuracy_metrics": accuracy_metrics,
            "feature_importance": feature_importance,
            "prediction_statistics": {
                "predictions_this_month": len(recent_predictions),
                "high_risk_predictions": len([p for p in recent_predictions if p.get("risk_score", 0) > 0.7]),
                "medium_risk_predictions": len([p for p in recent_predictions if 0.3 <= p.get("risk_score", 0) <= 0.7]),
                "low_risk_predictions": len([p for p in recent_predictions if p.get("risk_score", 0) < 0.3])
            }
        }
        
        return create_response(
            success=True,
            message="Model performance metrics retrieved successfully",
            data=performance_data
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model performance: {str(e)}"
        )


@router.post("/model/train")
async def retrain_model(
    training_request: ModelRetrainRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Initiate model retraining with new data (Admin only)
    """
    try:
        # Validate training request
        start_date = training_request.data_source_period["start_date"]
        end_date = training_request.data_source_period["end_date"]
        
        if start_date >= end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date range for training data"
            )
        
        # Create training job record
        training_job_id = generate_id("TRAINING_")
        training_job = {
            "id": training_job_id,
            "model_type": training_request.model_type,
            "requested_by": current_user["uid"],
            "priority": training_request.priority,
            "status": "queued",
            "data_source_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "hyperparameters": training_request.hyperparameters or {},
            "validation_method": training_request.validation_method,
            "created_at": datetime.now(timezone.utc),
            "progress": 0
        }
        
        # Store training job
        await firestore_service.create_document(
            "model_training_jobs",
            training_job,
            training_job_id
        )
        
        # Queue background training task
        background_tasks.add_task(
            execute_model_training,
            training_job_id,
            training_request,
            current_user["uid"]
        )
        
        return create_response(
            success=True,
            message="Model training job queued successfully",
            data={
                "training_job_id": training_job_id,
                "status": "queued",
                "estimated_completion": "1-2 hours",
                "priority": training_request.priority
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue model training: {str(e)}"
        )


async def execute_model_training(training_job_id: str, training_request: ModelRetrainRequest, requested_by: str):
    """
    Background task to execute model training
    """
    try:
        # Update job status
        await firestore_service.update_document(
            "model_training_jobs",
            training_job_id,
            {
                "status": "running",
                "started_at": datetime.now(timezone.utc),
                "progress": 10
            }
        )
        
        # Collect training data
        start_date = training_request.data_source_period["start_date"]
        end_date = training_request.data_source_period["end_date"]
        
        # Get water quality data
        water_data = await firestore_service.query_collection(
            FirestoreCollections.WATER_QUALITY,
            filters=[
                ("collection_time", ">=", start_date),
                ("collection_time", "<=", end_date)
            ],
            limit=10000
        )
        
        # Get health data
        health_data = await firestore_service.query_collection(
            FirestoreCollections.HEALTH_DATA,
            filters=[
                ("report_time", ">=", start_date),
                ("report_time", "<=", end_date)
            ],
            limit=10000
        )
        
        # Update progress
        await firestore_service.update_document(
            "model_training_jobs",
            training_job_id,
            {"progress": 30}
        )
        
        # Prepare training dataset
        training_data = []
        labels = []
        
        # This is a simplified example - actual implementation would be more sophisticated
        for i, water_record in enumerate(water_data[:1000]):  # Limit for demo
            # Create synthetic labels based on water quality
            ph = water_record.get('ph_level', 7.0)
            turbidity = water_record.get('turbidity', 0.0)
            
            # Simple rule-based labeling for demonstration
            risk_label = 0.0
            if ph < 6.5 or ph > 8.5:
                risk_label += 0.4
            if turbidity > 5.0:
                risk_label += 0.3
            
            # Add health context if available
            nearby_health = [h for h in health_data if h.get('location') == water_record.get('location')]
            if len(nearby_health) > 2:
                risk_label += 0.3
            
            risk_label = min(1.0, risk_label)
            
            # Create training sample
            training_sample = {
                'location': water_record.get('location', {}),
                'water_quality_data': {
                    'ph_level': ph,
                    'turbidity': turbidity,
                    'residual_chlorine': water_record.get('residual_chlorine', 0.0),
                    'temperature': water_record.get('temperature', 25.0)
                },
                'health_symptoms_count': {},
                'population_density': 100,
                'sanitation_score': 5.0
            }
            
            training_data.append(training_sample)
            labels.append(risk_label)
        
        # Update progress
        await firestore_service.update_document(
            "model_training_jobs",
            training_job_id,
            {"progress": 60}
        )
        
        # Train model
        if len(training_data) > 10:  # Minimum data required
            training_results = disease_prediction_model.train_model(training_data, labels)
            
            # Update progress
            await firestore_service.update_document(
                "model_training_jobs",
                training_job_id,
                {
                    "progress": 90,
                    "training_results": training_results
                }
            )
            
            # Complete training job
            await firestore_service.update_document(
                "model_training_jobs",
                training_job_id,
                {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc),
                    "progress": 100,
                    "final_results": training_results
                }
            )
        else:
            # Insufficient data
            await firestore_service.update_document(
                "model_training_jobs",
                training_job_id,
                {
                    "status": "failed",
                    "error": "Insufficient training data",
                    "completed_at": datetime.now(timezone.utc)
                }
            )
        
    except Exception as e:
        # Mark job as failed
        await firestore_service.update_document(
            "model_training_jobs",
            training_job_id,
            {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now(timezone.utc)
            }
        )


@router.get("/model/training-jobs")
async def get_training_jobs(
    limit: int = Query(20, le=100),
    status: Optional[str] = Query(None, pattern="^(queued|running|completed|failed)$"),
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Get model training job history (Admin only)
    """
    try:
        filters = []
        if status:
            filters.append(("status", "==", status))
        
        # Query training jobs
        training_jobs = await firestore_service.query_collection(
            "model_training_jobs",
            filters=filters,
            order_by="created_at",
            limit=limit
        )
        
        # Sort by created_at descending
        training_jobs.sort(key=lambda x: x.get("created_at"), reverse=True)
        
        return create_response(
            success=True,
            message="Training jobs retrieved successfully",
            data=training_jobs,
            meta={"count": len(training_jobs), "limit": limit}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve training jobs: {str(e)}"
        )


@router.get("/model/training-jobs/{job_id}")
async def get_training_job_details(
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Get detailed information about a specific training job
    """
    try:
        training_job = await firestore_service.get_document(
            "model_training_jobs",
            job_id
        )
        
        if not training_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training job not found"
            )
        
        return create_response(
            success=True,
            message="Training job details retrieved successfully",
            data=training_job
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve training job details: {str(e)}"
        )


@router.post("/validate-prediction")
async def validate_prediction(
    prediction_id: str,
    actual_outcome: bool,
    notes: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Validate a prediction with actual outcome for model improvement
    """
    try:
        # Get the original prediction
        prediction = await firestore_service.get_document(
            FirestoreCollections.ML_PREDICTIONS,
            prediction_id
        )
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found"
            )
        
        # Calculate accuracy
        predicted_risk = prediction.get("risk_score", 0.5)
        predicted_outcome = predicted_risk > 0.5
        accuracy_score = 1.0 if predicted_outcome == actual_outcome else 0.0
        
        # Update prediction with validation
        validation_data = {
            "validated": True,
            "actual_outcome": actual_outcome,
            "validation_accuracy": accuracy_score,
            "validated_by": current_user["uid"],
            "validated_at": datetime.now(timezone.utc),
            "validation_notes": notes
        }
        
        await firestore_service.update_document(
            FirestoreCollections.ML_PREDICTIONS,
            prediction_id,
            validation_data
        )
        
        # Store validation for model improvement
        validation_record = {
            "id": generate_id("VALIDATION_"),
            "prediction_id": prediction_id,
            "predicted_risk_score": predicted_risk,
            "actual_outcome": actual_outcome,
            "accuracy_score": accuracy_score,
            "validated_by": current_user["uid"],
            "validation_date": datetime.now(timezone.utc),
            "notes": notes
        }
        
        await firestore_service.create_document(
            "prediction_validations",
            validation_record
        )
        
        return create_response(
            success=True,
            message="Prediction validation recorded successfully",
            data={
                "prediction_id": prediction_id,
                "accuracy_score": accuracy_score,
                "validated_at": validation_data["validated_at"].isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate prediction: {str(e)}"
        )


@router.get("/analytics/model-accuracy")
async def get_model_accuracy_analytics(
    time_period: str = Query("30d", pattern="^(7d|30d|90d|1y)$"),
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Get model accuracy analytics over time
    """
    try:
        from datetime import timedelta
        
        # Calculate date range
        days_map = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}
        days = days_map[time_period]
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get validated predictions
        validated_predictions = await firestore_service.query_collection(
            "prediction_validations",
            filters=[("validation_date", ">=", start_date)],
            limit=1000
        )
        
        if not validated_predictions:
            return create_response(
                success=True,
                message="No validated predictions found for the specified period",
                data={
                    "time_period": time_period,
                    "total_validations": 0,
                    "overall_accuracy": 0.0,
                    "accuracy_by_risk_level": {}
                }
            )
        
        # Calculate overall accuracy
        total_validations = len(validated_predictions)
        accurate_predictions = sum(1 for v in validated_predictions if v.get("accuracy_score", 0) == 1.0)
        overall_accuracy = accurate_predictions / total_validations if total_validations > 0 else 0.0
        
        # Calculate accuracy by risk level
        accuracy_by_risk = {"high": [], "medium": [], "low": []}
        
        for validation in validated_predictions:
            risk_score = validation.get("predicted_risk_score", 0.5)
            accuracy = validation.get("accuracy_score", 0)
            
            if risk_score > 0.7:
                accuracy_by_risk["high"].append(accuracy)
            elif risk_score > 0.3:
                accuracy_by_risk["medium"].append(accuracy)
            else:
                accuracy_by_risk["low"].append(accuracy)
        
        # Calculate averages
        accuracy_summary = {}
        for level, scores in accuracy_by_risk.items():
            if scores:
                accuracy_summary[level] = {
                    "accuracy": sum(scores) / len(scores),
                    "count": len(scores)
                }
            else:
                accuracy_summary[level] = {"accuracy": 0.0, "count": 0}
        
        analytics_data = {
            "time_period": time_period,
            "total_validations": total_validations,
            "overall_accuracy": round(overall_accuracy, 3),
            "accuracy_by_risk_level": accuracy_summary,
            "improvement_suggestions": []
        }
        
        # Add improvement suggestions based on accuracy
        if overall_accuracy < 0.7:
            analytics_data["improvement_suggestions"].append("Model accuracy below 70% - consider retraining with recent data")
        if accuracy_summary["high"]["accuracy"] < accuracy_summary["low"]["accuracy"]:
            analytics_data["improvement_suggestions"].append("High-risk predictions less accurate than low-risk - review high-risk thresholds")
        
        return create_response(
            success=True,
            message="Model accuracy analytics retrieved successfully",
            data=analytics_data
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model accuracy analytics: {str(e)}"
        )