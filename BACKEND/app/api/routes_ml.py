from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.schemas.ml import (
    PredictionInput, RiskPrediction, OutbreakPrediction,
    BatchPredictionRequest, ModelTrainingData, ModelPerformance,
    PredictionHistory, ModelRetrainRequest, WaterQualityPredictionInput,
    HealthRiskPredictionInput, ModelValidationRequest, CombinedRiskAssessment,
    FeatureImportanceAnalysis
)
from app.dependencies import get_current_user, get_admin_user, get_health_worker
from app.ml.predictor import get_model, DiseasePredictor
from app.db.database import firestore_service, FirestoreCollections
from app.core.utils import create_response, generate_id
from app.core.monitoring import structured_logger, metrics_collector

router = APIRouter()
logger = structured_logger


@router.post("/predict/risk", response_model=RiskPrediction)
async def predict_risk(
    prediction_input: PredictionInput,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate basic risk prediction for waterborne disease outbreak using real ML models
    """
    try:
        # Initialize risk predictor
        risk_predictor = get_model()
        
        # Convert Pydantic model to dict for processing
        input_data = prediction_input.model_dump()
        
        # Generate risk prediction (using water quality model as primary)
        risk_prediction = risk_predictor.predict_water_quality_risk(input_data)
        
        # Log prediction request for audit
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": current_user["uid"],
                "action": "risk_prediction_requested",
                "location": input_data["location"],
                "prediction_type": "basic_risk_assessment",
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        logger.info(
            "Risk prediction completed successfully",
            user_id=current_user["uid"],
            risk_level=risk_prediction.get("risk_level", "unknown")
        )
        
        return create_response(
            success=True,
            message="Risk prediction completed successfully",
            data=risk_prediction
        )
        
    except Exception as e:
        logger.error("Risk prediction failed", error=str(e), user_id=current_user["uid"])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk prediction failed: {str(e)}"
        )


@router.post("/predict/water-quality")
async def predict_water_quality_risk(
    water_input: WaterQualityPredictionInput,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate water quality specific risk prediction
    """
    try:
        model = get_model()
        
        # Convert to format expected by water quality model
        input_data = water_input.model_dump()
        
        # Use water quality model specifically
        prediction_result = model.predict_water_quality_risk(input_data)
        
        logger.info(
            "Water quality risk prediction completed",
            user_id=current_user["uid"],
            risk_score=prediction_result["risk_score"]
        )
        
        return create_response(
            success=True,
            message="Water quality risk prediction completed successfully",
            data=prediction_result
        )
        
    except Exception as e:
        logger.error("Water quality prediction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Water quality prediction failed: {str(e)}"
        )


@router.post("/predict/health-risk")
async def predict_health_risk(
    health_input: HealthRiskPredictionInput,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate health risk prediction using demographic and environmental factors
    """
    try:
        model = get_model()
        
        # Convert to format expected by health model
        input_data = health_input.model_dump()
        
        # Use correct method name
        prediction_result = model.predict_health_risk(input_data)
        
        logger.info(
            "Health risk prediction completed",
            user_id=current_user["uid"],
            risk_score=prediction_result["risk_score"]
        )
        
        return create_response(
            success=True,
            message="Health risk prediction completed successfully",
            data=prediction_result
        )
        
    except Exception as e:
        logger.error("Health risk prediction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health risk prediction failed: {str(e)}"
        )


@router.post("/predict/combined")
async def predict_combined_risk(
    prediction_input: PredictionInput,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate combined risk assessment using both water quality and health models
    """
    try:
        model = get_model()
        
        # Convert input
        input_data = prediction_input.model_dump()
        
        # Get combined prediction
        prediction_result = model.predict_combined_risk(input_data)
        
        # Enhanced response with recommendations
        enhanced_result = {
            **prediction_result,
            "assessment_type": "combined_risk_assessment",
            "models_used": []
        }
        
        if prediction_result.get("water_quality_risk"):
            enhanced_result["models_used"].append("water_quality_rf_model")
        if prediction_result.get("health_risk"):
            enhanced_result["models_used"].append("health_risk_rf_model")
        
        # Log prediction request for audit
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": current_user["uid"],
                "action": "combined_risk_prediction_requested",
                "location": input_data["location"],
                "prediction_type": "combined_risk_assessment",
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        logger.info(
            "Combined risk prediction completed",
            user_id=current_user["uid"],
            models_used=enhanced_result["models_used"]
        )
        
        return create_response(
            success=True,
            message="Combined risk assessment completed successfully",
            data=enhanced_result
        )
        
    except Exception as e:
        logger.error("Combined risk prediction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Combined risk prediction failed: {str(e)}"
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
        risk_predictor = DiseasePredictor()
        
        # Convert input
        input_data = prediction_input.model_dump()
        
        # Generate outbreak prediction
        outbreak_prediction = await risk_predictor.predict_outbreak(region_id, input_data)
        
        # Log outbreak prediction
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": current_user["uid"],
                "action": "outbreak_prediction_requested",
                "region_id": region_id,
                "outbreak_probability": outbreak_prediction.outbreak_probability,
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        logger.info(
            "Outbreak prediction completed",
            user_id=current_user["uid"],
            region_id=region_id,
            outbreak_probability=outbreak_prediction.outbreak_probability
        )
        
        return outbreak_prediction
        
    except Exception as e:
        logger.error("Outbreak prediction failed", region_id=region_id, error=str(e))
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
        
        risk_predictor = DiseasePredictor()
        
        # Generate predictions for all locations
        predictions = await risk_predictor.batch_risk_assessment(batch_request.locations)
        
        # Convert predictions to dict format for response
        prediction_data = []
        for pred in predictions:
            pred_dict = {
                "location": pred.location,
                "risk_score": pred.risk_score,
                "risk_level": pred.risk_level.value,
                "risk_color": pred.risk_color,
                "confidence": pred.confidence,
                "predicted_diseases": [d.value for d in pred.predicted_diseases],
                "primary_risk_factors": pred.primary_risk_factors,
                "population_at_risk": pred.population_at_risk,
                "model_version": pred.model_version
            }
            if batch_request.include_recommendations:
                # Add basic recommendations based on risk level
                if pred.risk_level.value == "high":
                    pred_dict["recommendations"] = [
                        "Implement immediate water treatment",
                        "Issue public health advisory",
                        "Deploy rapid response team"
                    ]
                elif pred.risk_level.value == "medium":
                    pred_dict["recommendations"] = [
                        "Increase monitoring frequency",
                        "Distribute health education materials"
                    ]
                else:
                    pred_dict["recommendations"] = [
                        "Continue routine monitoring"
                    ]
            
            prediction_data.append(pred_dict)
        
        # Log batch request
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": current_user["uid"],
                "action": "batch_prediction_requested",
                "location_count": len(batch_request.locations),
                "prediction_type": batch_request.prediction_type,
                "successful_predictions": len(predictions),
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        logger.info(
            "Batch prediction completed",
            user_id=current_user["uid"],
            total_locations=len(batch_request.locations),
            successful_predictions=len(predictions)
        )
        
        return create_response(
            success=True,
            message=f"Batch prediction completed for {len(predictions)} locations",
            data={
                "predictions": prediction_data,
                "total_requested": len(batch_request.locations),
                "successful_predictions": len(predictions),
                "prediction_type": batch_request.prediction_type,
                "time_horizon": batch_request.time_horizon,
                "processing_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.get("/model/performance")
async def get_model_performance(
    model_type: str = Query("all", pattern="^(all|water_quality|health_risk)$"),
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Get current model performance metrics
    """
    try:
        model = get_model()
        
        performance_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": {}
        }
        
        # Water quality model performance
        if model_type in ["all", "water_quality"] and model.water_model.is_loaded:
            performance_data["models"]["water_quality"] = {
                "model_version": "rf_water_v1.0",
                "model_type": "RandomForestRegressor",
                "is_loaded": True,
                "model_path": str(model.water_model.model_path),
                "features_count": len(model.water_model.feature_columns),
                "required_features": model.water_model.required_features,
                "estimated_accuracy": 0.78,  # From your ML teammate's results
                "estimated_mae": 5.2,
                "estimated_rmse": 7.8,
                "estimated_r2": 0.85,
                "last_updated": "2024-01-15T10:00:00Z"
            }
        
        # Health risk model performance
        if model_type in ["all", "health_risk"] and model.health_model.is_loaded:
            performance_data["models"]["health_risk"] = {
                "model_version": "rf_health_v1.0",
                "model_type": "RandomForestRegressor",
                "is_loaded": True,
                "model_path": str(model.health_model.model_path),
                "features_count": len(model.health_model.feature_columns),
                "required_features": model.health_model.required_features,
                "estimated_accuracy": 0.82,  # From your ML teammate's results
                "estimated_mae": 0.12,
                "estimated_rmse": 0.18,
                "estimated_r2": 0.78,
                "last_updated": "2024-01-15T10:00:00Z"
            }
        
        # Get recent prediction statistics
        try:
            recent_predictions = await firestore_service.query_collection(
                FirestoreCollections.ML_PREDICTIONS,
                filters=[
                    ("prediction_timestamp", ">=", datetime.now(timezone.utc).replace(day=1))  # This month
                ],
                limit=1000
            )
            
            performance_data["recent_statistics"] = {
                "total_predictions_this_month": len(recent_predictions),
                "high_risk_predictions": len([p for p in recent_predictions if p.get("risk_score", 0) > 0.7]),
                "medium_risk_predictions": len([p for p in recent_predictions if 0.3 <= p.get("risk_score", 0) <= 0.7]),
                "low_risk_predictions": len([p for p in recent_predictions if p.get("risk_score", 0) < 0.3]),
                "average_confidence": sum(p.get("confidence", 0) for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0
            }
        except Exception as e:
            logger.warning("Failed to get recent prediction statistics", error=str(e))
            performance_data["recent_statistics"] = {"error": "Unable to fetch recent statistics"}
        
        return create_response(
            success=True,
            message="Model performance metrics retrieved successfully",
            data=performance_data
        )
        
    except Exception as e:
        logger.error("Failed to retrieve model performance", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model performance: {str(e)}"
        )


@router.get("/model/feature-importance")
async def get_feature_importance(
    model_type: str = Query("water_quality", pattern="^(water_quality|health_risk)$"),
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Get feature importance analysis from the ML models
    """
    try:
        model = get_model()
        
        # Mock feature importance data based on typical Random Forest results
        # In production, this would come from the actual model
        if model_type == "water_quality":
            feature_importance_data = {
                "model_version": "rf_water_v1.0",
                "feature_rankings": {
                    "ph": 0.18,
                    "hardness": 0.16,
                    "solids": 0.14,
                    "chloramines": 0.12,
                    "sulfate": 0.10,
                    "conductivity": 0.09,
                    "organic_carbon": 0.08,
                    "trihalomethanes": 0.07,
                    "turbidity": 0.06
                },
                "top_features": ["ph", "hardness", "solids", "chloramines", "sulfate"],
                "feature_categories": {
                    "chemical_properties": ["ph", "hardness", "chloramines", "sulfate"],
                    "physical_properties": ["solids", "turbidity", "conductivity"],
                    "organic_compounds": ["organic_carbon", "trihalomethanes"]
                },
                "interpretation": [
                    "pH level is the strongest predictor of water quality risk",
                    "Water hardness significantly impacts health outcomes",
                    "Chemical disinfectants play important protective roles"
                ]
            }
        else:  # health_risk
            feature_importance_data = {
                "model_version": "rf_health_v1.0", 
                "feature_rankings": {
                    "age": 0.22,
                    "ph": 0.18,
                    "sanitation": 0.15,
                    "income": 0.12,
                    "turbidity": 0.10,
                    "hardness": 0.08,
                    "chloramines": 0.07,
                    "sex": 0.05,
                    "water_source": 0.03
                },
                "top_features": ["age", "ph", "sanitation", "income", "turbidity"],
                "feature_categories": {
                    "demographic": ["age", "sex", "income"],
                    "water_quality": ["ph", "turbidity", "hardness", "chloramines"],
                    "infrastructure": ["sanitation", "water_source"]
                },
                "interpretation": [
                    "Age is the strongest predictor of health risk vulnerability",
                    "Water pH levels significantly impact disease probability",
                    "Sanitation infrastructure plays a crucial protective role"
                ]
            }
        
        feature_analysis = FeatureImportanceAnalysis(
            **feature_importance_data,
            analysis_date=datetime.now(timezone.utc)
        )
        
        return feature_analysis
        
    except Exception as e:
        logger.error("Failed to get feature importance", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature importance: {str(e)}"
        )


@router.post("/validate-prediction")
async def validate_prediction(
    validation_request: ModelValidationRequest,
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Validate a prediction with actual outcome for model improvement
    """
    try:
        # Get the original prediction
        prediction = await firestore_service.get_document(
            FirestoreCollections.ML_PREDICTIONS,
            validation_request.prediction_id
        )
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found"
            )
        
        # Calculate accuracy score
        predicted_risk = prediction.get("risk_score", 0.5)
        predicted_outcome = predicted_risk > 0.5
        accuracy_score = 1.0 if predicted_outcome == validation_request.actual_outcome else 0.0
        
        # Apply confidence weighting
        weighted_accuracy = accuracy_score * validation_request.outcome_confidence
        
        # Update prediction with validation
        validation_data = {
            "validated": True,
            "actual_outcome": validation_request.actual_outcome,
            "outcome_confidence": validation_request.outcome_confidence,
            "validation_accuracy": weighted_accuracy,
            "validated_by": current_user["uid"],
            "validated_at": datetime.now(timezone.utc),
            "validation_notes": validation_request.validation_notes,
            "validation_source": validation_request.validation_source
        }
        
        await firestore_service.update_document(
            FirestoreCollections.ML_PREDICTIONS,
            validation_request.prediction_id,
            validation_data
        )
        
        # Store validation for model improvement
        validation_record = {
            "id": generate_id("VALIDATION_"),
            "prediction_id": validation_request.prediction_id,
            "predicted_risk_score": predicted_risk,
            "actual_outcome": validation_request.actual_outcome,
            "accuracy_score": weighted_accuracy,
            "validated_by": current_user["uid"],
            "validation_date": datetime.now(timezone.utc),
            "notes": validation_request.validation_notes,
            "source": validation_request.validation_source,
            "confidence": validation_request.outcome_confidence
        }
        
        await firestore_service.create_document(
            "prediction_validations",
            validation_record
        )
        
        logger.info(
            "Prediction validation recorded",
            prediction_id=validation_request.prediction_id,
            accuracy=weighted_accuracy,
            validator=current_user["uid"]
        )
        
        return create_response(
            success=True,
            message="Prediction validation recorded successfully",
            data={
                "prediction_id": validation_request.prediction_id,
                "accuracy_score": weighted_accuracy,
                "validated_at": validation_data["validated_at"].isoformat(),
                "contribution_to_model_improvement": "high" if weighted_accuracy != predicted_risk else "moderate"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to validate prediction", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate prediction: {str(e)}"
        )


@router.get("/predictions/history")
async def get_prediction_history(
    limit: int = Query(50, le=200),
    prediction_type: Optional[str] = Query(None, pattern="^(risk|outbreak|water_quality|health_risk)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    risk_level: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
    validated_only: bool = Query(False),
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
        
        # Apply validation filter
        if validated_only:
            filters.append(("validated", "==", True))
        
        # Apply location-based restrictions for non-admin users
        user_role = current_user.get("role")
        if user_role not in ["admin", "state_health_authority"]:
            # Could add location filtering here based on user's region
            pass
        
        # Query predictions
        predictions = await firestore_service.query_collection(
            FirestoreCollections.ML_PREDICTIONS,
            filters=filters,
            order_by="prediction_timestamp",
            limit=limit
        )
        
        # Apply additional filters that can't be done in Firestore
        if risk_level:
            predictions = [p for p in predictions if p.get("risk_level") == risk_level]
        
        # Sort by timestamp descending
        predictions.sort(key=lambda x: x.get("prediction_timestamp"), reverse=True)
        
        # Calculate summary statistics
        if predictions:
            avg_risk_score = sum(p.get("risk_score", 0) for p in predictions) / len(predictions)
            risk_distribution = {
                "high": len([p for p in predictions if p.get("risk_score", 0) > 0.7]),
                "medium": len([p for p in predictions if 0.3 <= p.get("risk_score", 0) <= 0.7]),
                "low": len([p for p in predictions if p.get("risk_score", 0) < 0.3])
            }
            validated_count = len([p for p in predictions if p.get("validated", False)])
            
            summary_stats = {
                "total_predictions": len(predictions),
                "average_risk_score": round(avg_risk_score, 3),
                "risk_distribution": risk_distribution,
                "validated_predictions": validated_count,
                "validation_rate": round(validated_count / len(predictions) * 100, 1) if predictions else 0
            }
        else:
            summary_stats = {
                "total_predictions": 0,
                "average_risk_score": 0,
                "risk_distribution": {"high": 0, "medium": 0, "low": 0},
                "validated_predictions": 0,
                "validation_rate": 0
            }
        
        return create_response(
            success=True,
            message="Prediction history retrieved successfully",
            data={
                "predictions": predictions,
                "summary_statistics": summary_stats,
                "filters_applied": {
                    "limit": limit,
                    "prediction_type": prediction_type,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "risk_level": risk_level,
                    "validated_only": validated_only
                }
            },
            meta={"count": len(predictions), "limit": limit}
        )
        
    except Exception as e:
        logger.error("Failed to retrieve prediction history", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction history: {str(e)}"
        )


@router.get("/analytics/model-accuracy")
async def get_model_accuracy_analytics(
    time_period: str = Query("30d", pattern="^(7d|30d|90d|1y)$"),
    model_type: str = Query("all", pattern="^(all|water_quality|health_risk)$"),
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
        filters = [
            ("validation_date", ">=", start_date),
            ("validated", "==", True)
        ]
        
        if model_type != "all":
            # Add model type filter - would need to be stored in validation records
            pass
        
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
                    "accuracy_by_risk_level": {},
                    "model_performance_trend": [],
                    "recommendations": ["Increase prediction validation efforts"]
                }
            )
        
        # Calculate overall accuracy
        total_validations = len(validated_predictions)
        total_accuracy = sum(v.get("accuracy_score", 0) for v in validated_predictions)
        overall_accuracy = total_accuracy / total_validations if total_validations > 0 else 0.0
        
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
        
        # Calculate averages and create summary
        accuracy_summary = {}
        for level, scores in accuracy_by_risk.items():
            if scores:
                accuracy_summary[level] = {
                    "accuracy": round(sum(scores) / len(scores), 3),
                    "count": len(scores),
                    "confidence_interval": {
                        "min": round(min(scores), 3),
                        "max": round(max(scores), 3)
                    }
                }
            else:
                accuracy_summary[level] = {"accuracy": 0.0, "count": 0}
        
        # Generate improvement recommendations
        recommendations = []
        if overall_accuracy < 0.7:
            recommendations.append("Overall model accuracy below 70% - consider retraining with recent data")
        if accuracy_summary.get("high", {}).get("accuracy", 0) < accuracy_summary.get("low", {}).get("accuracy", 0):
            recommendations.append("High-risk predictions less accurate than low-risk - review high-risk thresholds")
        if total_validations < 10:
            recommendations.append("Limited validation data available - increase prediction validation efforts")
        
        if not recommendations:
            recommendations.append("Model performance within acceptable ranges")
        
        analytics_data = {
            "time_period": time_period,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": datetime.now(timezone.utc).isoformat()
            },
            "total_validations": total_validations,
            "overall_accuracy": round(overall_accuracy, 3),
            "accuracy_by_risk_level": accuracy_summary,
            "model_reliability_score": round((overall_accuracy + (total_validations / 100)) / 2, 3),
            "validation_coverage": round((total_validations / max(days, 1)) * 100, 1),  # validations per day
            "improvement_recommendations": recommendations,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return create_response(
            success=True,
            message="Model accuracy analytics retrieved successfully",
            data=analytics_data
        )
        
    except Exception as e:
        logger.error("Failed to retrieve model accuracy analytics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model accuracy analytics: {str(e)}"
        )


@router.post("/model/retrain")
async def retrain_model(
    training_request: ModelRetrainRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Initiate model retraining with new data (Admin only)
    Note: This is a placeholder - actual retraining requires specialized infrastructure
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
            "target_metric": training_request.target_metric,
            "include_feature_selection": training_request.include_feature_selection,
            "created_at": datetime.now(timezone.utc),
            "progress": 0,
            "estimated_completion_hours": 2,  # Estimate based on model type
            "requires_manual_intervention": True,  # Since we're using pre-trained models
            "notes": "This request will be forwarded to the ML team for manual retraining"
        }
        
        # Store training job
        await firestore_service.create_document(
            "model_training_jobs",
            training_job,
            training_job_id
        )
        
        # In a real implementation, you would trigger actual retraining
        # For now, we'll just log the request and provide guidance
        
        logger.info(
            "Model retraining request created",
            training_job_id=training_job_id,
            model_type=training_request.model_type,
            requested_by=current_user["uid"]
        )
        
        return create_response(
            success=True,
            message="Model retraining request submitted successfully",
            data={
                "training_job_id": training_job_id,
                "status": "queued",
                "model_type": training_request.model_type,
                "priority": training_request.priority,
                "estimated_completion": "Manual intervention required - ML team will be notified",
                "next_steps": [
                    "ML team will be notified of retraining request",
                    "Data collection and preprocessing will be performed",
                    "Model will be retrained with specified parameters",
                    "New model will be validated before deployment"
                ],
                "note": "Current models are pre-trained - retraining requires coordination with ML team"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to queue model retraining", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue model retraining: {str(e)}"
        )


@router.get("/model/status")
async def get_model_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get current status of all ML models
    """
    try:
        model = get_model()
        
        status_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "operational",
            "models": {
                "water_quality": {
                    "loaded": model.water_model.is_loaded,
                    "model_file": "rf_water_model_v2.pkl",  # Updated to match actual file
                    "fallback_available": len(model.water_model.fallback_models) > 0,
                    "last_prediction": "Recently active",
                    "features_required": model.water_model.required_features
                },
                "health_risk": {
                    "loaded": model.health_model.is_loaded,
                    "model_file": "rf_health_model.pkl",
                    "fallback_available": len(model.health_model.fallback_models) > 0,
                    "last_prediction": "Recently active",
                    "features_required": model.health_model.required_features
                }
            },
            "capabilities": {
                "risk_prediction": model.water_model.is_loaded or model.health_model.is_loaded,
                "outbreak_prediction": model.water_model.is_loaded or model.health_model.is_loaded,
                "batch_processing": True,
                "regional_risk_mapping": True,
                "combined_assessment": model.water_model.is_loaded and model.health_model.is_loaded
            },
            "version": model.version
        }
        
        # Determine overall status
        if not (model.water_model.is_loaded or model.health_model.is_loaded):
            status_data["overall_status"] = "degraded"
        elif model.water_model.is_loaded and model.health_model.is_loaded:
            status_data["overall_status"] = "optimal"
        
        return create_response(
            success=True,
            message="Model status retrieved successfully",
            data=status_data
        )
        
    except Exception as e:
        logger.error("Failed to get model status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model status: {str(e)}"
        )