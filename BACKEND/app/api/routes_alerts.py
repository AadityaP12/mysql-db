from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, get_health_worker, get_admin_user
from app.db.database import firestore_service, FirestoreCollections
from app.core.utils import (
    create_response, generate_id, get_language_text, 
    format_phone_number, ALERT_MESSAGES, get_risk_level
)

router = APIRouter()


class AlertCreate(BaseModel):
    alert_type: str = Field(..., pattern="^(water_contamination|disease_outbreak|high_risk|maintenance|general)$")
    severity: str = Field(..., pattern="^(low|medium|high|critical)$")
    title: str = Field(..., min_length=5, max_length=200)
    message: str = Field(..., min_length=10, max_length=1000)
    location: Dict[str, float]
    affected_radius_km: float = Field(..., ge=0.1, le=50)
    target_audience: List[str] = Field(..., description="Roles to send alert to")
    languages: List[str] = Field(default=["en"], description="Languages for multilingual alerts")
    expiry_time: Optional[datetime] = None
    action_required: bool = False
    contact_info: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_type": "water_contamination",
                "severity": "high",
                "title": "Water Contamination Detected",
                "message": "High levels of bacterial contamination detected in local water source. Please boil water before consumption.",
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "affected_radius_km": 2.0,
                "target_audience": ["user", "health_worker"],
                "languages": ["en", "as", "hi"],
                "action_required": True,
                "contact_info": "Contact local health center: +91-XXXXXXXXXX"
            }
        }


class AlertUpdate(BaseModel):
    status: Optional[str] = Field(None, pattern="^(active|resolved|expired|cancelled)$")
    message: Optional[str] = None
    severity: Optional[str] = Field(None, pattern="^(low|medium|high|critical)$")
    expiry_time: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "resolved",
                "message": "Water contamination issue has been resolved. Water is now safe for consumption."
            }
        }


class EmergencyAlertCreate(BaseModel):
    location: Dict[str, float]
    description: str = Field(..., min_length=10, max_length=500)
    contact_number: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": {"latitude": 26.1445, "longitude": 91.7362},
                "description": "Medical emergency - patient showing severe symptoms of waterborne disease",
                "contact_number": "+91-9876543210"
            }
        }


class NotificationPreferences(BaseModel):
    user_uid: str
    sms_enabled: bool = True
    email_enabled: bool = True
    push_enabled: bool = True
    alert_types: List[str] = Field(default=["water_contamination", "disease_outbreak", "high_risk"])
    max_radius_km: float = Field(default=10.0, ge=0.1, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_uid": "firebase_user_id",
                "sms_enabled": True,
                "email_enabled": True,
                "push_enabled": True,
                "alert_types": ["water_contamination", "disease_outbreak"],
                "max_radius_km": 5.0
            }
        }


@router.post("/create")
async def create_alert(
    alert_data: AlertCreate,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_health_worker)
):
    """
    Create a new alert and send notifications
    """
    try:
        # Generate alert ID
        alert_id = generate_id("ALERT_")
        
        # Prepare alert record
        alert_record = alert_data.dict()
        alert_record.update({
            "id": alert_id,
            "created_by": current_user["uid"],
            "created_by_role": current_user.get("role"),
            "created_at": datetime.now(timezone.utc),
            "status": "active",
            "region": current_user.get("region"),
            "state": current_user.get("state"),
            "district": current_user.get("district"),
            "recipients_count": 0,
            "read_count": 0,
            "response_count": 0
        })
        
        # Set default expiry time if not provided
        if not alert_record["expiry_time"]:
            # Default expiry based on severity
            hours_map = {"low": 72, "medium": 48, "high": 24, "critical": 12}
            hours = hours_map.get(alert_data.severity, 24)
            alert_record["expiry_time"] = datetime.now(timezone.utc) + timedelta(hours=hours)
        
        # Store alert in Firestore
        await firestore_service.create_document(
            FirestoreCollections.ALERTS,
            alert_record,
            alert_id
        )
        
        # Queue background task to send notifications
        background_tasks.add_task(
            send_alert_notifications,
            alert_id,
            alert_data,
            current_user
        )
        
        return create_response(
            success=True,
            message="Alert created and notifications queued successfully",
            data={
                "alert_id": alert_id,
                "status": "active",
                "created_at": alert_record["created_at"]
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alert: {str(e)}"
        )


async def send_alert_notifications(
    alert_id: str,
    alert_data: AlertCreate,
    creator: Dict[str, Any]
):
    """
    Background task to send alert notifications to users
    """
    try:
        from app.core.utils import calculate_distance
        
        # Get all users who should receive this alert
        users = await firestore_service.query_collection(
            FirestoreCollections.USERS,
            filters=[("active", "==", True)]
        )
        
        # Get notification preferences for users
        preferences = await firestore_service.query_collection(
            FirestoreCollections.USERS,  # Assuming preferences are in user profiles
            filters=[]
        )
        
        recipients = []
        
        for user in users:
            user_role = user.get("role", "user")
            
            # Check if user role is in target audience
            if user_role not in alert_data.target_audience:
                continue
            
            # Check if user is within affected radius (if location available)
            user_location = user.get("location")
            if user_location and "latitude" in user_location:
                distance = calculate_distance(
                    alert_data.location["latitude"],
                    alert_data.location["longitude"],
                    user_location["latitude"],
                    user_location["longitude"]
                )
                
                if distance > alert_data.affected_radius_km:
                    continue
            
            # Get user's preferred language
            user_language = user.get("preferred_language", "en")
            
            # Prepare notification
            notification_data = {
                "id": generate_id("NOTIF_"),
                "alert_id": alert_id,
                "user_uid": user["uid"],
                "alert_type": alert_data.alert_type,
                "severity": alert_data.severity,
                "title": alert_data.title,
                "message": get_language_text(
                    ALERT_MESSAGES.get(user_language, ALERT_MESSAGES["en"]).get(
                        alert_data.alert_type, alert_data.message
                    ),
                    user_language
                ) if alert_data.alert_type in ALERT_MESSAGES.get(user_language, {}) else alert_data.message,
                "location": alert_data.location,
                "created_at": datetime.now(timezone.utc),
                "read": False,
                "delivery_status": "pending",
                "language": user_language
            }
            
            # Store notification
            await firestore_service.create_document(
                "notifications",  # Create notifications collection
                notification_data
            )
            
            recipients.append(user["uid"])
            
            # Send SMS if enabled (mock implementation)
            if user.get("sms_enabled", True) and alert_data.severity in ["high", "critical"]:
                await send_sms_alert(user, notification_data)
        
        # Update alert with recipient count
        await firestore_service.update_document(
            FirestoreCollections.ALERTS,
            alert_id,
            {
                "recipients_count": len(recipients),
                "notification_sent_at": datetime.now(timezone.utc)
            }
        )
        
    except Exception as e:
        # Log error but don't fail the alert creation
        print(f"Error sending notifications for alert {alert_id}: {str(e)}")


async def send_sms_alert(user: Dict[str, Any], notification: Dict[str, Any]):
    """
    Send SMS alert (mock implementation - integrate with SMS service)
    """
    try:
        phone_number = format_phone_number(user.get("phone_number", ""))
        message = f"HEALTH ALERT: {notification['title']} - {notification['message']}"
        
        # Mock SMS sending - replace with actual SMS service integration
        print(f"SMS Alert sent to {phone_number}: {message}")
        
        # Update notification delivery status
        await firestore_service.update_document(
            "notifications",
            notification["id"],
            {
                "delivery_status": "sent",
                "sms_sent_at": datetime.now(timezone.utc)
            }
        )
        
    except Exception as e:
        print(f"Failed to send SMS to {user.get('phone_number')}: {str(e)}")


@router.get("/")
async def get_alerts(
    status: Optional[str] = Query(None, pattern="^(active|resolved|expired|cancelled)$"),
    alert_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    limit: int = Query(50, le=200),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get alerts with filters
    """
    try:
        filters = []
        
        # Apply status filter
        if status:
            filters.append(("status", "==", status))
        else:
            # Default to active and recent alerts
            filters.append(("status", "in", ["active", "resolved"]))
        
        # Apply type and severity filters
        if alert_type:
            filters.append(("alert_type", "==", alert_type))
        if severity:
            filters.append(("severity", "==", severity))
        
        # Apply location-based filtering for non-admin users
        user_role = current_user.get("role")
        if user_role not in ["admin", "state_health_authority"]:
            if current_user.get("state"):
                filters.append(("state", "==", current_user.get("state")))
            if current_user.get("district"):
                filters.append(("district", "==", current_user.get("district")))
        
        # Query alerts
        alerts = await firestore_service.query_collection(
            FirestoreCollections.ALERTS,
            filters=filters,
            order_by="created_at",
            limit=limit
        )
        
        # Sort by created_at descending (most recent first)
        alerts.sort(key=lambda x: x.get("created_at"), reverse=True)
        
        return create_response(
            success=True,
            message="Alerts retrieved successfully",
            data=alerts,
            meta={"count": len(alerts), "limit": limit}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )


@router.get("/{alert_id}")
async def get_alert_details(
    alert_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get detailed information about a specific alert
    """
    try:
        # Get alert
        alert = await firestore_service.get_document(
            FirestoreCollections.ALERTS,
            alert_id
        )
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # Check if user has access to this alert
        user_role = current_user.get("role")
        if user_role not in ["admin", "state_health_authority"]:
            # Check if alert is for user's region
            if (current_user.get("state") != alert.get("state") or
                current_user.get("district") != alert.get("district")):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this alert"
                )
        
        # Get notification statistics if user is creator or admin
        if (current_user["uid"] == alert.get("created_by") or 
            user_role in ["admin", "state_health_authority"]):
            
            # Get notification stats
            notifications = await firestore_service.query_collection(
                "notifications",
                filters=[("alert_id", "==", alert_id)]
            )
            
            stats = {
                "total_sent": len(notifications),
                "read_count": len([n for n in notifications if n.get("read", False)]),
                "delivery_success": len([n for n in notifications if n.get("delivery_status") == "sent"]),
                "delivery_pending": len([n for n in notifications if n.get("delivery_status") == "pending"])
            }
            
            alert["notification_stats"] = stats
        
        return create_response(
            success=True,
            message="Alert details retrieved successfully",
            data=alert
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alert details: {str(e)}"
        )


@router.put("/{alert_id}")
async def update_alert(
    alert_id: str,
    alert_update: AlertUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update an alert (creator or admin only)
    """
    try:
        # Get existing alert
        alert = await firestore_service.get_document(
            FirestoreCollections.ALERTS,
            alert_id
        )
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # Check permissions
        user_role = current_user.get("role")
        if (current_user["uid"] != alert.get("created_by") and 
            user_role not in ["admin", "state_health_authority"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only alert creator or admin can update alerts"
            )
        
        # Prepare update data
        update_data = {}
        for field, value in alert_update.dict(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value
        
        if update_data:
            update_data.update({
                "updated_at": datetime.now(timezone.utc),
                "updated_by": current_user["uid"]
            })
            
            # Update alert
            await firestore_service.update_document(
                FirestoreCollections.ALERTS,
                alert_id,
                update_data
            )
            
            # If status changed to resolved/cancelled, update notifications
            if "status" in update_data and update_data["status"] in ["resolved", "cancelled"]:
                await firestore_service.query_collection(
                    "notifications",
                    filters=[("alert_id", "==", alert_id)]
                )
                # Update all related notifications - implement batch update if needed
        
        return create_response(
            success=True,
            message="Alert updated successfully",
            data={"alert_id": alert_id, "updated_fields": list(update_data.keys())}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update alert: {str(e)}"
        )


@router.get("/user/notifications")
async def get_user_notifications(
    unread_only: bool = Query(False),
    limit: int = Query(50, le=200),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get notifications for current user
    """
    try:
        filters = [("user_uid", "==", current_user["uid"])]
        
        if unread_only:
            filters.append(("read", "==", False))
        
        notifications = await firestore_service.query_collection(
            "notifications",
            filters=filters,
            order_by="created_at",
            limit=limit
        )
        
        # Sort by created_at descending
        notifications.sort(key=lambda x: x.get("created_at"), reverse=True)
        
        return create_response(
            success=True,
            message="Notifications retrieved successfully",
            data=notifications,
            meta={
                "count": len(notifications),
                "limit": limit,
                "unread_count": len([n for n in notifications if not n.get("read", False)])
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve notifications: {str(e)}"
        )


@router.put("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Mark a notification as read
    """
    try:
        # Get notification
        notification = await firestore_service.get_document(
            "notifications",
            notification_id
        )
        
        if not notification:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found"
            )
        
        # Check if notification belongs to current user
        if notification.get("user_uid") != current_user["uid"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this notification"
            )
        
        # Mark as read
        await firestore_service.update_document(
            "notifications",
            notification_id,
            {
                "read": True,
                "read_at": datetime.now(timezone.utc)
            }
        )
        
        # Update alert read count
        alert_id = notification.get("alert_id")
        if alert_id:
            alert = await firestore_service.get_document(
                FirestoreCollections.ALERTS,
                alert_id
            )
            if alert:
                new_read_count = alert.get("read_count", 0) + 1
                await firestore_service.update_document(
                    FirestoreCollections.ALERTS,
                    alert_id,
                    {"read_count": new_read_count}
                )
        
        return create_response(
            success=True,
            message="Notification marked as read"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark notification as read: {str(e)}"
        )


@router.post("/notifications/mark-all-read")
async def mark_all_notifications_read(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Mark all unread notifications as read for current user
    """
    try:
        # Get all unread notifications for user
        unread_notifications = await firestore_service.query_collection(
            "notifications",
            filters=[
                ("user_uid", "==", current_user["uid"]),
                ("read", "==", False)
            ]
        )
        
        # Batch update all to read
        if unread_notifications:
            batch_operations = []
            for notification in unread_notifications:
                batch_operations.append({
                    "action": "update",
                    "collection": "notifications",
                    "doc_id": notification["id"],
                    "data": {
                        "read": True,
                        "read_at": datetime.now(timezone.utc)
                    }
                })
            
            await firestore_service.batch_write(batch_operations)
        
        return create_response(
            success=True,
            message=f"Marked {len(unread_notifications)} notifications as read",
            data={"marked_count": len(unread_notifications)}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark notifications as read: {str(e)}"
        )


@router.post("/emergency")
async def create_emergency_alert(
    emergency_data: EmergencyAlertCreate,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create an emergency alert (any user can create)
    """
    try:
        # Create emergency alert
        alert_data = AlertCreate(
            alert_type="general",
            severity="critical",
            title="Emergency Alert",
            message=f"Emergency reported: {emergency_data.description}",
            location=emergency_data.location,
            affected_radius_km=5.0,
            target_audience=["health_worker", "asha", "local_authority", "admin"],
            languages=["en"],
            action_required=True,
            contact_info=emergency_data.contact_number
        )
        
        # Generate alert ID
        alert_id = generate_id("EMERGENCY_")
        
        # Prepare alert record
        alert_record = alert_data.dict()
        alert_record.update({
            "id": alert_id,
            "created_by": current_user["uid"],
            "created_by_role": current_user.get("role"),
            "created_at": datetime.now(timezone.utc),
            "status": "active",
            "emergency": True,
            "region": current_user.get("region"),
            "state": current_user.get("state"),
            "district": current_user.get("district"),
            "expiry_time": datetime.now(timezone.utc) + timedelta(hours=6),  # 6 hours for emergency
            "recipients_count": 0
        })
        
        # Store alert
        await firestore_service.create_document(
            FirestoreCollections.ALERTS,
            alert_record,
            alert_id
        )
        
        # Send immediate notifications
        background_tasks.add_task(
            send_alert_notifications,
            alert_id,
            alert_data,
            current_user
        )
        
        return create_response(
            success=True,
            message="Emergency alert created successfully. Authorities have been notified.",
            data={
                "alert_id": alert_id,
                "status": "active",
                "emergency": True
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create emergency alert: {str(e)}"
        )


@router.get("/statistics")
async def get_alert_statistics(
    time_period: str = Query("30d", pattern="^(7d|30d|90d|1y)$"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get alert statistics for dashboard
    """
    try:
        # Calculate date range
        days_map = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}
        days = days_map[time_period]
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Build location filters
        location_filters = []
        user_role = current_user.get("role")
        
        if user_role not in ["admin", "state_health_authority"]:
            if current_user.get("state"):
                location_filters.append(("state", "==", current_user.get("state")))
            if current_user.get("district"):
                location_filters.append(("district", "==", current_user.get("district")))
        
        # Get alerts in time period
        alerts = await firestore_service.query_collection(
            FirestoreCollections.ALERTS,
            filters=[("created_at", ">=", start_date)] + location_filters
        )
        
        # Calculate statistics
        total_alerts = len(alerts)
        active_alerts = len([a for a in alerts if a.get("status") == "active"])
        resolved_alerts = len([a for a in alerts if a.get("status") == "resolved"])
        
        # Count by type
        alert_types = {}
        for alert in alerts:
            alert_type = alert.get("alert_type", "unknown")
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for alert in alerts:
            severity = alert.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Average response time (for resolved alerts)
        response_times = []
        for alert in alerts:
            if alert.get("status") == "resolved" and alert.get("updated_at"):
                created_at = alert.get("created_at")
                updated_at = alert.get("updated_at")
                if created_at and updated_at:
                    # Convert to comparable format if needed
                    response_time_hours = (updated_at - created_at).total_seconds() / 3600
                    response_times.append(response_time_hours)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        statistics = {
            "time_period": time_period,
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": resolved_alerts,
            "alert_types": alert_types,
            "severity_distribution": severity_counts,
            "average_response_time_hours": round(avg_response_time, 2),
            "resolution_rate": round((resolved_alerts / total_alerts * 100) if total_alerts > 0 else 0, 2)
        }
        
        return create_response(
            success=True,
            message="Alert statistics retrieved successfully",
            data=statistics
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alert statistics: {str(e)}"
        )


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: str,
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Delete an alert (Admin only)
    """
    try:
        # Get alert
        alert = await firestore_service.get_document(
            FirestoreCollections.ALERTS,
            alert_id
        )
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # Soft delete - mark as deleted
        await firestore_service.update_document(
            FirestoreCollections.ALERTS,
            alert_id,
            {
                "deleted": True,
                "deleted_at": datetime.now(timezone.utc),
                "deleted_by": current_user["uid"],
                "status": "cancelled"
            }
        )
        
        # Cancel all related notifications
        notifications = await firestore_service.query_collection(
            "notifications",
            filters=[("alert_id", "==", alert_id)]
        )
        
        if notifications:
            batch_operations = []
            for notification in notifications:
                batch_operations.append({
                    "action": "update",
                    "collection": "notifications",
                    "doc_id": notification["id"],
                    "data": {
                        "cancelled": True,
                        "cancelled_at": datetime.now(timezone.utc)
                    }
                })
            
            await firestore_service.batch_write(batch_operations)
        
        # Log deletion
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "action": "alert_deleted",
                "alert_id": alert_id,
                "alert_type": alert.get("alert_type"),
                "deleted_by": current_user["uid"],
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        return create_response(
            success=True,
            message="Alert deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete alert: {str(e)}"
        )


@router.post("/preferences")
async def update_notification_preferences(
    preferences: NotificationPreferences,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update notification preferences for a user
    """
    try:
        # Ensure user can only update their own preferences
        if preferences.user_uid != current_user["uid"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only update your own notification preferences"
            )
        
        # Update user profile with notification preferences
        await firestore_service.update_document(
            FirestoreCollections.USERS,
            current_user["uid"],
            {
                "notification_preferences": preferences.dict(),
                "preferences_updated_at": datetime.now(timezone.utc)
            }
        )
        
        return create_response(
            success=True,
            message="Notification preferences updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update notification preferences: {str(e)}"
        )