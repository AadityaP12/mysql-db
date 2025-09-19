from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.schemas.data import (
    WaterQualityData, HouseholdData, HealthData, SensorData,
    DataUploadResponse, DataQuery, BulkDataUpload
)
from app.dependencies import get_current_user, get_data_collector, get_health_worker
from app.db.database import firestore_service, FirestoreCollections
from app.core.utils import create_response, generate_id, validate_coordinates
from app.ml.predictor import RiskPredictor

router = APIRouter()
risk_predictor = RiskPredictor()


@router.post("/water-quality", response_model=DataUploadResponse)
async def upload_water_quality_data(
    water_data: WaterQualityData,
    current_user: Dict[str, Any] = Depends(get_data_collector)
):
    """
    Upload water quality data from IoT sensors or manual testing
    """
    try:
        # Validate coordinates
        lat, lon = water_data.location["latitude"], water_data.location["longitude"]
        if not validate_coordinates(lat, lon):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid coordinates for Northeast India region"
            )
        
        # Generate unique ID
        data_id = generate_id("WQ_")
        
        # Prepare data for storage
        water_quality_record = water_data.dict()
        water_quality_record.update({
            "id": data_id,
            "uploaded_by": current_user["uid"],
            "upload_timestamp": datetime.now(timezone.utc),
            "region": current_user.get("region"),
            "state": current_user.get("state"),
            "district": current_user.get("district"),
            "processed": False,
            "quality_status": "pending_analysis"
        })
        
        # Store in Firestore
        await firestore_service.create_document(
            FirestoreCollections.WATER_QUALITY,
            water_quality_record,
            data_id
        )
        
        # Trigger risk assessment if needed
        if water_data.ph_level < 6.5 or water_data.ph_level > 8.5 or water_data.turbidity > 5:
            water_quality_record["quality_status"] = "requires_attention"
            await firestore_service.update_document(
                FirestoreCollections.WATER_QUALITY,
                data_id,
                {"quality_status": "requires_attention"}
            )
        
        return DataUploadResponse(
            success=True,
            message="Water quality data uploaded successfully",
            data_id=data_id,
            processed_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload water quality data: {str(e)}"
        )


@router.post("/health", response_model=DataUploadResponse)
async def upload_health_data(
    health_data: HealthData,
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Upload health/symptom data by health workers
    """
    try:
        # Validate coordinates
        lat, lon = health_data.location["latitude"], health_data.location["longitude"]
        if not validate_coordinates(lat, lon):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid coordinates for Northeast India region"
            )
        
        # Generate unique ID
        data_id = generate_id("HD_")
        
        # Prepare data for storage
        health_record = health_data.dict()
        health_record.update({
            "id": data_id,
            "reported_by": current_user["uid"],
            "reporter_role": current_user.get("role"),
            "upload_timestamp": datetime.now(timezone.utc),
            "region": current_user.get("region"),
            "state": current_user.get("state"),
            "district": current_user.get("district"),
            "processed": False,
            "risk_assessed": False
        })
        
        # Store in Firestore
        await firestore_service.create_document(
            FirestoreCollections.HEALTH_DATA,
            health_record,
            data_id
        )
        
        # Check for high-risk symptoms
        high_risk_symptoms = ["severe_diarrhea", "bloody_stool", "severe_dehydration", "high_fever"]
        if any(symptom in health_data.symptoms for symptom in high_risk_symptoms):
            # Create immediate alert
            alert_data = {
                "id": generate_id("ALERT_"),
                "type": "health_emergency",
                "severity": "high",
                "patient_name": health_data.patient_name,
                "location": health_data.location,
                "symptoms": health_data.symptoms,
                "reported_by": current_user["uid"],
                "timestamp": datetime.now(timezone.utc),
                "status": "active"
            }
            
            await firestore_service.create_document(
                FirestoreCollections.ALERTS,
                alert_data
            )
        
        return DataUploadResponse(
            success=True,
            message="Health data uploaded successfully",
            data_id=data_id,
            processed_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload health data: {str(e)}"
        )


@router.post("/household", response_model=DataUploadResponse)
async def upload_household_data(
    household_data: HouseholdData,
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Upload household registration data
    """
    try:
        # Check if household already exists
        existing_household = await firestore_service.query_collection(
            FirestoreCollections.HOUSEHOLDS,
            filters=[("house_id", "==", household_data.house_id)]
        )
        
        if existing_household:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Household with this ID already exists"
            )
        
        # Prepare data for storage
        household_record = household_data.dict()
        household_record.update({
            "registered_by": current_user["uid"],
            "registration_timestamp": datetime.now(timezone.utc),
            "region": current_user.get("region"),
            "state": current_user.get("state"),
            "district": current_user.get("district"),
            "active": True
        })
        
        # Store in Firestore
        await firestore_service.create_document(
            FirestoreCollections.HOUSEHOLDS,
            household_record,
            household_data.house_id
        )
        
        return DataUploadResponse(
            success=True,
            message="Household data registered successfully",
            data_id=household_data.house_id,
            processed_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload household data: {str(e)}"
        )


@router.post("/sensor", response_model=DataUploadResponse)
async def upload_sensor_data(
    sensor_data: SensorData,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload IoT sensor data
    """
    try:
        # Validate sensor permissions
        if current_user.get("role") not in ["iot_sensor", "admin", "health_worker"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for sensor data upload"
            )
        
        # Generate unique ID
        data_id = generate_id("SENSOR_")
        
        # Prepare data for storage
        sensor_record = sensor_data.dict()
        sensor_record.update({
            "id": data_id,
            "uploaded_by": current_user["uid"],
            "upload_timestamp": datetime.now(timezone.utc),
            "processed": False
        })
        
        # Store in Firestore
        await firestore_service.create_document(
            FirestoreCollections.IOT_SENSORS,
            sensor_record,
            data_id
        )
        
        # Process water quality sensor data
        if sensor_data.sensor_type == "water_quality":
            # Convert to water quality format
            water_quality_data = {
                "sensor_id": sensor_data.sensor_id,
                "location": sensor_data.location,
                "ph_level": sensor_data.readings.get("ph", 7.0),
                "turbidity": sensor_data.readings.get("turbidity", 0.0),
                "residual_chlorine": sensor_data.readings.get("residual_chlorine"),
                "temperature": sensor_data.readings.get("temperature"),
                "collection_method": "iot_sensor",
                "collected_by": current_user["uid"],
                "collection_time": sensor_data.timestamp,
                "processed": False
            }
            
            # Store as water quality data too
            wq_id = generate_id("WQ_SENSOR_")
            await firestore_service.create_document(
                FirestoreCollections.WATER_QUALITY,
                water_quality_data,
                wq_id
            )
        
        return DataUploadResponse(
            success=True,
            message="Sensor data uploaded successfully",
            data_id=data_id,
            processed_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload sensor data: {str(e)}"
        )


@router.post("/bulk-upload")
async def bulk_upload_data(
    bulk_data: BulkDataUpload,
    current_user: Dict[str, Any] = Depends(get_data_collector)
):
    """
    Bulk upload multiple data records
    """
    try:
        uploaded_ids = []
        failed_records = []
        
        for idx, record in enumerate(bulk_data.data_records):
            try:
                data_id = generate_id(f"{bulk_data.data_type.upper()}_")
                
                # Add metadata to each record
                record.update({
                    "id": data_id,
                    "uploaded_by": current_user["uid"],
                    "upload_timestamp": datetime.now(timezone.utc),
                    "upload_source": bulk_data.upload_source,
                    "batch_upload": True,
                    "processed": False
                })
                
                # Determine collection based on data type
                collection_map = {
                    "health": FirestoreCollections.HEALTH_DATA,
                    "water_quality": FirestoreCollections.WATER_QUALITY,
                    "household": FirestoreCollections.HOUSEHOLDS,
                    "sensor": FirestoreCollections.IOT_SENSORS
                }
                
                collection = collection_map.get(bulk_data.data_type)
                if not collection:
                    raise ValueError(f"Invalid data type: {bulk_data.data_type}")
                
                # Store record
                await firestore_service.create_document(collection, record, data_id)
                uploaded_ids.append(data_id)
                
            except Exception as e:
                failed_records.append({
                    "index": idx,
                    "error": str(e),
                    "record": record
                })
        
        return create_response(
            success=len(failed_records) == 0,
            message=f"Bulk upload completed. {len(uploaded_ids)} successful, {len(failed_records)} failed",
            data={
                "uploaded_ids": uploaded_ids,
                "failed_records": failed_records,
                "total_processed": len(bulk_data.data_records)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk upload failed: {str(e)}"
        )


@router.get("/water-quality")
async def get_water_quality_data(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    limit: int = Query(50, le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get water quality data with filters
    """
    try:
        filters = []
        
        # Apply date filters
        if start_date:
            filters.append(("collection_time", ">=", start_date))
        if end_date:
            filters.append(("collection_time", "<=", end_date))
        
        # Apply location filters based on user role
        user_role = current_user.get("role")
        if user_role not in ["admin", "state_health_authority"]:
            # Restrict to user's region
            if current_user.get("state"):
                filters.append(("state", "==", current_user.get("state")))
            if current_user.get("district"):
                filters.append(("district", "==", current_user.get("district")))
        else:
            # Admin can filter by any state/district
            if state:
                filters.append(("state", "==", state))
            if district:
                filters.append(("district", "==", district))
        
        # Query data
        water_quality_data = await firestore_service.query_collection(
            FirestoreCollections.WATER_QUALITY,
            filters=filters,
            order_by="collection_time",
            limit=limit
        )
        
        return create_response(
            success=True,
            message="Water quality data retrieved successfully",
            data=water_quality_data,
            meta={"count": len(water_quality_data), "limit": limit}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve water quality data: {str(e)}"
        )


@router.get("/health")
async def get_health_data(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    symptoms: Optional[str] = Query(None, description="Comma-separated list of symptoms"),
    severity: Optional[str] = Query(None),
    limit: int = Query(50, le=1000),
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Get health data with filters (Health workers and above only)
    """
    try:
        filters = []
        
        # Apply date filters
        if start_date:
            filters.append(("report_time", ">=", start_date))
        if end_date:
            filters.append(("report_time", "<=", end_date))
        
        # Apply symptom filter
        if symptoms:
            symptom_list = [s.strip() for s in symptoms.split(",")]
            filters.append(("symptoms", "array_contains_any", symptom_list))
        
        # Apply severity filter
        if severity:
            filters.append(("symptom_severity", "==", severity))
        
        # Apply location-based restrictions
        user_role = current_user.get("role")
        if user_role not in ["admin", "state_health_authority"]:
            if current_user.get("state"):
                filters.append(("state", "==", current_user.get("state")))
            if current_user.get("district"):
                filters.append(("district", "==", current_user.get("district")))
        
        # Query data
        health_data = await firestore_service.query_collection(
            FirestoreCollections.HEALTH_DATA,
            filters=filters,
            order_by="report_time",
            limit=limit
        )
        
        # Remove sensitive patient information for non-health workers
        if user_role not in ["health_worker", "asha", "admin"]:
            for record in health_data:
                record.pop("patient_name", None)
        
        return create_response(
            success=True,
            message="Health data retrieved successfully",
            data=health_data,
            meta={"count": len(health_data), "limit": limit}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve health data: {str(e)}"
        )


@router.get("/households")
async def get_household_data(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    water_source: Optional[str] = Query(None),
    limit: int = Query(50, le=1000),
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Get household data with filters
    """
    try:
        filters = []
        
        # Apply filters
        if water_source:
            filters.append(("water_source", "==", water_source))
        
        # Apply location-based restrictions
        user_role = current_user.get("role")
        if user_role not in ["admin", "state_health_authority"]:
            if current_user.get("state"):
                filters.append(("state", "==", current_user.get("state")))
            if current_user.get("district"):
                filters.append(("district", "==", current_user.get("district")))
        else:
            if state:
                filters.append(("state", "==", state))
            if district:
                filters.append(("district", "==", district))
        
        # Query data
        household_data = await firestore_service.query_collection(
            FirestoreCollections.HOUSEHOLDS,
            filters=filters,
            order_by="registration_date",
            limit=limit
        )
        
        return create_response(
            success=True,
            message="Household data retrieved successfully",
            data=household_data,
            meta={"count": len(household_data), "limit": limit}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve household data: {str(e)}"
        )


@router.get("/analytics/summary")
async def get_data_analytics_summary(
    time_period: str = Query("7d", pattern="^(1d|7d|30d|90d)$"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get analytics summary for dashboard
    """
    try:
        from datetime import timedelta
        
        # Calculate date range
        days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
        days = days_map[time_period]
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Build location filters based on user role
        location_filters = []
        user_role = current_user.get("role")
        
        if user_role not in ["admin", "state_health_authority"]:
            if current_user.get("state"):
                location_filters.append(("state", "==", current_user.get("state")))
            if current_user.get("district"):
                location_filters.append(("district", "==", current_user.get("district")))
        
        # Get water quality data count
        wq_filters = [("collection_time", ">=", start_date)] + location_filters
        water_quality_count = len(await firestore_service.query_collection(
            FirestoreCollections.WATER_QUALITY,
            filters=wq_filters
        ))
        
        # Get health data count
        health_filters = [("report_time", ">=", start_date)] + location_filters
        health_reports_count = len(await firestore_service.query_collection(
            FirestoreCollections.HEALTH_DATA,
            filters=health_filters
        ))
        
        # Get active alerts count
        alert_filters = [("timestamp", ">=", start_date), ("status", "==", "active")] + location_filters
        active_alerts_count = len(await firestore_service.query_collection(
            FirestoreCollections.ALERTS,
            filters=alert_filters
        ))
        
        # Get household count
        household_count = len(await firestore_service.query_collection(
            FirestoreCollections.HOUSEHOLDS,
            filters=location_filters
        ))
        
        # Get recent high-risk water sources
        high_risk_water = await firestore_service.query_collection(
            FirestoreCollections.WATER_QUALITY,
            filters=[("quality_status", "==", "requires_attention")] + location_filters,
            limit=10
        )
        
        # Get common symptoms
        recent_health_data = await firestore_service.query_collection(
            FirestoreCollections.HEALTH_DATA,
            filters=health_filters,
            limit=100
        )
        
        # Count symptoms
        symptom_counts = {}
        for record in recent_health_data:
            for symptom in record.get("symptoms", []):
                symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
        
        # Sort symptoms by frequency
        common_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        analytics_data = {
            "time_period": time_period,
            "summary": {
                "water_quality_tests": water_quality_count,
                "health_reports": health_reports_count,
                "active_alerts": active_alerts_count,
                "registered_households": household_count
            },
            "high_risk_water_sources": len(high_risk_water),
            "common_symptoms": [{"symptom": s[0], "count": s[1]} for s in common_symptoms],
            "recent_water_quality_issues": high_risk_water[:5]  # Latest 5
        }
        
        return create_response(
            success=True,
            message="Analytics summary retrieved successfully",
            data=analytics_data
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics summary: {str(e)}"
        )


@router.delete("/data/{data_type}/{data_id}")
async def delete_data_record(
    data_type: str,
    data_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a data record (Admin only or original uploader)
    """
    try:
        # Validate data type
        collection_map = {
            "health": FirestoreCollections.HEALTH_DATA,
            "water_quality": FirestoreCollections.WATER_QUALITY,
            "household": FirestoreCollections.HOUSEHOLDS,
            "sensor": FirestoreCollections.IOT_SENSORS
        }
        
        collection = collection_map.get(data_type)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data type"
            )
        
        # Get the record to check ownership
        record = await firestore_service.get_document(collection, data_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Data record not found"
            )
        
        # Check permissions
        user_role = current_user.get("role")
        uploaded_by = record.get("uploaded_by") or record.get("reported_by") or record.get("registered_by")
        
        if user_role not in ["admin", "state_health_authority"] and uploaded_by != current_user["uid"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to delete this record"
            )
        
        # Soft delete - mark as deleted instead of actual deletion
        await firestore_service.update_document(
            collection,
            data_id,
            {
                "deleted": True,
                "deleted_at": datetime.now(timezone.utc),
                "deleted_by": current_user["uid"]
            }
        )
        
        # Log the deletion
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "action": "data_deleted",
                "data_type": data_type,
                "data_id": data_id,
                "deleted_by": current_user["uid"],
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        return create_response(
            success=True,
            message=f"{data_type.replace('_', ' ').title()} record deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete data record: {str(e)}"
        )


@router.get("/export/{data_type}")
async def export_data(
    data_type: str,
    format: str = Query("json", pattern="^(json|csv)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Export data in JSON or CSV format
    """
    try:
        # Check permissions
        user_role = current_user.get("role")
        if user_role not in ["admin", "state_health_authority", "local_authority"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for data export"
            )
        
        # Validate data type
        collection_map = {
            "health": FirestoreCollections.HEALTH_DATA,
            "water_quality": FirestoreCollections.WATER_QUALITY,
            "household": FirestoreCollections.HOUSEHOLDS,
            "sensor": FirestoreCollections.IOT_SENSORS
        }
        
        collection = collection_map.get(data_type)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data type"
            )
        
        # Build filters
        filters = []
        if start_date:
            date_field = "report_time" if data_type == "health" else "collection_time"
            if data_type == "household":
                date_field = "registration_date"
            filters.append((date_field, ">=", start_date))
        
        if end_date:
            date_field = "report_time" if data_type == "health" else "collection_time"
            if data_type == "household":
                date_field = "registration_date"
            filters.append((date_field, "<=", end_date))
        
        # Apply location restrictions
        if user_role not in ["admin", "state_health_authority"]:
            if current_user.get("state"):
                filters.append(("state", "==", current_user.get("state")))
            if current_user.get("district"):
                filters.append(("district", "==", current_user.get("district")))
        
        # Query data
        data = await firestore_service.query_collection(
            collection,
            filters=filters,
            limit=10000  # Large limit for export
        )
        
        if format == "csv":
            # Convert to CSV format
            import csv
            from io import StringIO
            
            if not data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No data found for export"
                )
            
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            
            from fastapi.responses import StreamingResponse
            
            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={data_type}_export.csv"
                }
            )
        
        else:  # JSON format
            return create_response(
                success=True,
                message=f"{data_type.replace('_', ' ').title()} data exported successfully",
                data={
                    "export_format": format,
                    "record_count": len(data),
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "records": data
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export data: {str(e)}"
        )