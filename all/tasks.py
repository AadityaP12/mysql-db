from celery import Celery
from celery.schedules import crontab
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import asyncio
import json
import time

from app.config import settings
from app.core.monitoring import structured_logger, metrics_collector, task_monitor
from app.core.cache import get_cache
from app.core.utils import calculate_distance, get_language_text, ALERT_MESSAGES
from app.db.database import firestore_service, FirestoreCollections

# Initialize Celery app
celery_app = Celery(
    "health_monitor_tasks",
    **settings.celery_config
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,
    task_routes={
        'app.tasks.send_alert_notifications': {'queue': 'alerts'},
        'app.tasks.process_ml_prediction': {'queue': 'ml'},
        'app.tasks.generate_daily_report': {'queue': 'reports'},
        'app.tasks.cleanup_old_data': {'queue': 'maintenance'},
        'app.tasks.backup_data': {'queue': 'maintenance'},
    },
    task_default_queue='default',
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

logger = structured_logger


@celery_app.task(bind=True, max_retries=3)
@task_monitor.track_task("send_alert_notifications")
def send_alert_notifications(self, alert_id: str, alert_data: Dict[str, Any], creator_uid: str):
    """
    Background task to send alert notifications to users
    """
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_alert_notifications_async(alert_id, alert_data, creator_uid))
        loop.close()
        return result
        
    except Exception as e:
        logger.error(
            "Alert notification task failed",
            alert_id=alert_id,
            error=str(e),
            retry_count=self.request.retries
        )
        
        if self.request.retries < 3:
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        else:
            # Mark alert as failed after all retries
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                firestore_service.update_document(
                    FirestoreCollections.ALERTS,
                    alert_id,
                    {
                        "notification_status": "failed",
                        "notification_error": str(e),
                        "failed_at": datetime.now(timezone.utc)
                    }
                )
            )
            loop.close()
            raise


async def _send_alert_notifications_async(alert_id: str, alert_data: Dict[str, Any], creator_uid: str):
    """Async implementation of alert notifications"""
    try:
        logger.info("Starting alert notification process", alert_id=alert_id)
        
        # Get all active users
        users = await firestore_service.query_collection(
            FirestoreCollections.USERS,
            filters=[("active", "==", True)]
        )
        
        recipients = []
        failed_recipients = []
        
        # Process each user
        for user in users:
            try:
                user_role = user.get("role", "user")
                
                # Check if user role is in target audience
                target_audience = alert_data.get("target_audience", [])
                if user_role not in target_audience:
                    continue
                
                # Check geographic proximity
                user_location = user.get("location")
                alert_location = alert_data.get("location")
                affected_radius = alert_data.get("affected_radius_km", 5.0)
                
                if user_location and alert_location:
                    if all(k in user_location for k in ["latitude", "longitude"]) and \
                       all(k in alert_location for k in ["latitude", "longitude"]):
                        
                        distance = calculate_distance(
                            alert_location["latitude"],
                            alert_location["longitude"],
                            user_location["latitude"],
                            user_location["longitude"]
                        )
                        
                        if distance > affected_radius:
                            continue
                
                # Create notification
                notification_id = f"notif_{int(time.time())}_{user['uid']}"
                user_language = user.get("preferred_language", "en")
                
                # Get localized message
                alert_type = alert_data.get("alert_type", "general")
                localized_messages = ALERT_MESSAGES.get(user_language, ALERT_MESSAGES["en"])
                message = localized_messages.get(alert_type, alert_data.get("message", ""))
                
                notification_data = {
                    "id": notification_id,
                    "alert_id": alert_id,
                    "user_uid": user["uid"],
                    "alert_type": alert_type,
                    "severity": alert_data.get("severity", "medium"),
                    "title": alert_data.get("title", "Health Alert"),
                    "message": message,
                    "location": alert_location,
                    "created_at": datetime.now(timezone.utc),
                    "read": False,
                    "delivery_status": "pending",
                    "language": user_language,
                    "channels": []
                }
                
                # Store notification in database
                await firestore_service.create_document(
                    "notifications",
                    notification_data,
                    notification_id
                )
                
                # Send via different channels based on severity and user preferences
                channels_used = []
                
                # SMS for high/critical severity
                if alert_data.get("severity") in ["high", "critical"]:
                    if user.get("phone_number") and user.get("sms_enabled", True):
                        sms_sent = await send_sms_notification(user, notification_data)
                        if sms_sent:
                            channels_used.append("sms")
                            metrics_collector.record_alert_sent(alert_type, alert_data.get("severity"), "sms")
                
                # Push notification (if enabled)
                if user.get("push_enabled", True):
                    push_sent = await send_push_notification(user, notification_data)
                    if push_sent:
                        channels_used.append("push")
                        metrics_collector.record_alert_sent(alert_type, alert_data.get("severity"), "push")
                
                # Email for important alerts
                if alert_data.get("severity") in ["high", "critical"] and user.get("email_enabled", True):
                    email_sent = await send_email_notification(user, notification_data)
                    if email_sent:
                        channels_used.append("email")
                        metrics_collector.record_alert_sent(alert_type, alert_data.get("severity"), "email")
                
                # Update notification with delivery status
                await firestore_service.update_document(
                    "notifications",
                    notification_id,
                    {
                        "delivery_status": "sent" if channels_used else "failed",
                        "channels": channels_used,
                        "delivered_at": datetime.now(timezone.utc)
                    }
                )
                
                recipients.append({
                    "user_uid": user["uid"],
                    "channels": channels_used,
                    "language": user_language
                })
                
            except Exception as e:
                logger.error(
                    "Failed to send notification to user",
                    user_uid=user.get("uid"),
                    error=str(e)
                )
                failed_recipients.append({
                    "user_uid": user.get("uid"),
                    "error": str(e)
                })
        
        # Update alert with notification results
        await firestore_service.update_document(
            FirestoreCollections.ALERTS,
            alert_id,
            {
                "recipients_count": len(recipients),
                "failed_recipients_count": len(failed_recipients),
                "notification_sent_at": datetime.now(timezone.utc),
                "notification_status": "completed"
            }
        )
        
        logger.info(
            "Alert notification process completed",
            alert_id=alert_id,
            successful_recipients=len(recipients),
            failed_recipients=len(failed_recipients)
        )
        
        return {
            "alert_id": alert_id,
            "recipients_count": len(recipients),
            "failed_count": len(failed_recipients),
            "channels_breakdown": _count_channels(recipients)
        }
        
    except Exception as e:
        logger.error("Alert notification process failed", alert_id=alert_id, error=str(e))
        raise


def _count_channels(recipients: List[Dict]) -> Dict[str, int]:
    """Count notifications sent by channel"""
    channel_counts = {"sms": 0, "push": 0, "email": 0}
    for recipient in recipients:
        for channel in recipient.get("channels", []):
            if channel in channel_counts:
                channel_counts[channel] += 1
    return channel_counts


async def send_sms_notification(user: Dict[str, Any], notification: Dict[str, Any]) -> bool:
    """Send SMS notification (mock implementation)"""
    try:
        phone_number = user.get("phone_number")
        if not phone_number:
            return False
        
        message = f"HEALTH ALERT: {notification['title']} - {notification['message'][:100]}"
        
        # Mock SMS sending - integrate with actual SMS service
        logger.info(
            "SMS notification sent",
            user_uid=user["uid"],
            phone_number=phone_number[:3] + "****" + phone_number[-3:],  # Masked for privacy
            message_length=len(message)
        )
        
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        return True
        
    except Exception as e:
        logger.error("SMS notification failed", user_uid=user.get("uid"), error=str(e))
        return False


async def send_push_notification(user: Dict[str, Any], notification: Dict[str, Any]) -> bool:
    """Send push notification (mock implementation)"""
    try:
        # Mock push notification - integrate with FCM or similar service
        logger.info(
            "Push notification sent",
            user_uid=user["uid"],
            title=notification["title"],
            message_length=len(notification["message"])
        )
        
        # Simulate API call delay
        await asyncio.sleep(0.05)
        
        return True
        
    except Exception as e:
        logger.error("Push notification failed", user_uid=user.get("uid"), error=str(e))
        return False


async def send_email_notification(user: Dict[str, Any], notification: Dict[str, Any]) -> bool:
    """Send email notification (mock implementation)"""
    try:
        email = user.get("email")
        if not email:
            return False
        
        # Mock email sending - integrate with email service
        logger.info(
            "Email notification sent",
            user_uid=user["uid"],
            email=email[:3] + "****@" + email.split("@")[1] if "@" in email else "masked",
            subject=notification["title"]
        )
        
        # Simulate API call delay
        await asyncio.sleep(0.2)
        
        return True
        
    except Exception as e:
        logger.error("Email notification failed", user_uid=user.get("uid"), error=str(e))
        return False


@celery_app.task(bind=True, max_retries=2)
@task_monitor.track_task("process_ml_prediction")
def process_ml_prediction(self, prediction_data: Dict[str, Any]):
    """
    Background task to process ML predictions and store results
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_process_ml_prediction_async(prediction_data))
        loop.close()
        return result
        
    except Exception as e:
        logger.error(
            "ML prediction task failed",
            prediction_id=prediction_data.get("id"),
            error=str(e),
            retry_count=self.request.retries
        )
        
        if self.request.retries < 2:
            raise self.retry(countdown=30 * (2 ** self.request.retries))
        else:
            raise


async def _process_ml_prediction_async(prediction_data: Dict[str, Any]):
    """Async implementation of ML prediction processing"""
    try:
        from app.ml.predictor import RiskPredictor
        from app.core.utils import generate_id
        
        predictor = RiskPredictor()
        
        # Generate prediction
        risk_prediction = await predictor.predict_risk(prediction_data)
        
        # Store prediction in database
        prediction_record = {
            "id": generate_id("PRED_"),
            "input_data": prediction_data,
            "risk_score": risk_prediction.risk_score,
            "risk_level": risk_prediction.risk_level,
            "confidence": risk_prediction.confidence,
            "predicted_diseases": risk_prediction.predicted_diseases,
            "primary_risk_factors": risk_prediction.primary_risk_factors,
            "model_version": risk_prediction.model_version,
            "prediction_timestamp": datetime.now(timezone.utc),
            "processed_at": datetime.now(timezone.utc)
        }
        
        await firestore_service.create_document(
            FirestoreCollections.ML_PREDICTIONS,
            prediction_record,
            prediction_record["id"]
        )
        
        # Record metrics
        metrics_collector.record_ml_prediction(
            "risk_assessment",
            risk_prediction.risk_level,
            risk_prediction.model_version
        )
        
        # If high risk, trigger alert
        if risk_prediction.risk_level == "HIGH":
            await _trigger_high_risk_alert(prediction_record)
        
        logger.info(
            "ML prediction processed successfully",
            prediction_id=prediction_record["id"],
            risk_level=risk_prediction.risk_level,
            risk_score=risk_prediction.risk_score
        )
        
        return {
            "prediction_id": prediction_record["id"],
            "risk_level": risk_prediction.risk_level,
            "risk_score": risk_prediction.risk_score
        }
        
    except Exception as e:
        logger.error("ML prediction processing failed", error=str(e))
        raise


async def _trigger_high_risk_alert(prediction_record: Dict[str, Any]):
    """Trigger alert for high-risk predictions"""
    try:
        from app.core.utils import generate_id
        
        location = prediction_record["input_data"].get("location", {})
        
        alert_data = {
            "id": generate_id("AUTO_ALERT_"),
            "alert_type": "high_risk",
            "severity": "high",
            "title": "High Risk Area Detected",
            "message": "ML model detected high risk of waterborne disease outbreak in this area. Please take preventive measures.",
            "location": location,
            "affected_radius_km": 5.0,
            "target_audience": ["health_worker", "asha", "local_authority"],
            "languages": ["en", "hi", "as"],
            "created_by": "system",
            "created_by_role": "system",
            "created_at": datetime.now(timezone.utc),
            "status": "active",
            "automated": True,
            "prediction_id": prediction_record["id"],
            "expiry_time": datetime.now(timezone.utc) + timedelta(hours=24)
        }
        
        # Store alert
        alert_id = await firestore_service.create_document(
            FirestoreCollections.ALERTS,
            alert_data,
            alert_data["id"]
        )
        
        # Trigger notification task
        send_alert_notifications.delay(alert_id, alert_data, "system")
        
        logger.info(
            "High risk alert triggered",
            alert_id=alert_id,
            prediction_id=prediction_record["id"]
        )
        
    except Exception as e:
        logger.error(
            "Failed to trigger high risk alert",
            prediction_id=prediction_record.get("id"),
            error=str(e)
        )


@celery_app.task(bind=True)
@task_monitor.track_task("generate_daily_report")
def generate_daily_report(self, date_str: str = None):
    """
    Generate daily health monitoring report
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_generate_daily_report_async(date_str))
        loop.close()
        return result
        
    except Exception as e:
        logger.error("Daily report generation failed", date=date_str, error=str(e))
        raise


async def _generate_daily_report_async(date_str: str = None):
    """Generate daily report for health monitoring"""
    try:
        from app.core.utils import generate_id, export_to_csv
        
        # Parse date or use today
        if date_str:
            report_date = datetime.fromisoformat(date_str)
        else:
            report_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        start_time = report_date
        end_time = report_date + timedelta(days=1)
        
        logger.info("Generating daily report", report_date=report_date.isoformat())
        
        # Collect data for the day
        health_data = await firestore_service.query_collection(
            FirestoreCollections.HEALTH_DATA,
            filters=[
                ("report_time", ">=", start_time),
                ("report_time", "<", end_time)
            ]
        )
        
        water_quality_data = await firestore_service.query_collection(
            FirestoreCollections.WATER_QUALITY,
            filters=[
                ("collection_time", ">=", start_time),
                ("collection_time", "<", end_time)
            ]
        )
        
        alerts_data = await firestore_service.query_collection(
            FirestoreCollections.ALERTS,
            filters=[
                ("created_at", ">=", start_time),
                ("created_at", "<", end_time)
            ]
        )
        
        predictions_data = await firestore_service.query_collection(
            FirestoreCollections.ML_PREDICTIONS,
            filters=[
                ("prediction_timestamp", ">=", start_time),
                ("prediction_timestamp", "<", end_time)
            ]
        )
        
        # Calculate statistics
        stats = {
            "date": report_date.isoformat(),
            "health_reports": len(health_data),
            "water_quality_tests": len(water_quality_data),
            "alerts_sent": len(alerts_data),
            "ml_predictions": len(predictions_data),
            "high_risk_predictions": len([p for p in predictions_data if p.get("risk_level") == "HIGH"]),
            "critical_alerts": len([a for a in alerts_data if a.get("severity") == "critical"]),
            "contaminated_sources": len([w for w in water_quality_data if w.get("quality_status") == "requires_attention"])
        }
        
        # Symptom analysis
        symptom_counts = {}
        for record in health_data:
            for symptom in record.get("symptoms", []):
                symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
        
        # State-wise breakdown
        state_breakdown = {}
        for record in health_data:
            state = record.get("state", "unknown")
            if state not in state_breakdown:
                state_breakdown[state] = {"health_reports": 0, "alerts": 0}
            state_breakdown[state]["health_reports"] += 1
        
        for alert in alerts_data:
            state = alert.get("state", "unknown")
            if state not in state_breakdown:
                state_breakdown[state] = {"health_reports": 0, "alerts": 0}
            state_breakdown[state]["alerts"] += 1
        
        # Create comprehensive report
        report_data = {
            "id": generate_id("REPORT_"),
            "type": "daily_summary",
            "date": report_date.isoformat(),
            "generated_at": datetime.now(timezone.utc),
            "summary_statistics": stats,
            "symptom_analysis": {
                "total_symptoms": len(symptom_counts),
                "most_common": sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            },
            "geographic_breakdown": state_breakdown,
            "data_quality": {
                "health_data_completeness": _calculate_data_completeness(health_data),
                "water_data_coverage": len(water_quality_data)
            },
            "recommendations": _generate_recommendations(stats, symptom_counts)
        }
        
        # Store report
        await firestore_service.create_document(
            "daily_reports",
            report_data,
            report_data["id"]
        )
        
        # Cache report for quick access
        cache = get_cache()
        await cache.set(
            f"daily_report:{report_date.strftime('%Y-%m-%d')}",
            report_data,
            ttl=86400  # 24 hours
        )
        
        logger.info(
            "Daily report generated successfully",
            report_id=report_data["id"],
            date=report_date.isoformat(),
            health_reports=stats["health_reports"],
            alerts_sent=stats["alerts_sent"]
        )
        
        return report_data
        
    except Exception as e:
        logger.error("Daily report generation failed", error=str(e))
        raise


def _calculate_data_completeness(health_data: List[Dict]) -> float:
    """Calculate data completeness percentage"""
    if not health_data:
        return 0.0
    
    required_fields = ["patient_name", "age", "symptoms", "location"]
    complete_records = 0
    
    for record in health_data:
        if all(field in record and record[field] for field in required_fields):
            complete_records += 1
    
    return (complete_records / len(health_data)) * 100


def _generate_recommendations(stats: Dict, symptom_counts: Dict) -> List[str]:
    """Generate recommendations based on daily statistics"""
    recommendations = []
    
    if stats["high_risk_predictions"] > 0:
        recommendations.append("High-risk areas detected. Increase monitoring and preventive measures.")
    
    if stats["contaminated_sources"] > 0:
        recommendations.append("Contaminated water sources found. Implement water treatment protocols.")
    
    if symptom_counts.get("diarrhea", 0) > 10:
        recommendations.append("High diarrhea reports. Consider waterborne disease investigation.")
    
    if stats["critical_alerts"] > 0:
        recommendations.append("Critical health alerts issued. Ensure emergency response protocols are active.")
    
    if not recommendations:
        recommendations.append("No critical issues detected. Continue routine monitoring.")
    
    return recommendations


@celery_app.task(bind=True)
@task_monitor.track_task("cleanup_old_data")
def cleanup_old_data(self):
    """
    Cleanup old data and expired records
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_cleanup_old_data_async())
        loop.close()
        return result
        
    except Exception as e:
        logger.error("Data cleanup task failed", error=str(e))
        raise


async def _cleanup_old_data_async():
    """Async implementation of data cleanup"""
    try:
        cleanup_results = {"expired_alerts": 0, "old_notifications": 0, "old_predictions": 0}
        
        # Expire old alerts
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        old_alerts = await firestore_service.query_collection(
            FirestoreCollections.ALERTS,
            filters=[
                ("status", "==", "active"),
                ("expiry_time", "<", cutoff_time)
            ]
        )
        
        for alert in old_alerts:
            await firestore_service.update_document(
                FirestoreCollections.ALERTS,
                alert["id"],
                {
                    "status": "expired",
                    "expired_at": datetime.now(timezone.utc)
                }
            )
            cleanup_results["expired_alerts"] += 1
        
        # Clean old notifications (older than 30 days)
        notification_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        old_notifications = await firestore_service.query_collection(
            "notifications",
            filters=[("created_at", "<", notification_cutoff)]
        )
        
        for notification in old_notifications:
            await firestore_service.update_document(
                "notifications",
                notification["id"],
                {
                    "deleted": True,
                    "deleted_at": datetime.now(timezone.utc)
                }
            )
            cleanup_results["old_notifications"] += 1
        
        # Clean old unvalidated predictions (older than 90 days)
        prediction_cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        old_predictions = await firestore_service.query_collection(
            FirestoreCollections.ML_PREDICTIONS,
            filters=[
                ("prediction_timestamp", "<", prediction_cutoff),
                ("validated", "==", False)
            ]
        )
        
        for prediction in old_predictions:
            await firestore_service.update_document(
                FirestoreCollections.ML_PREDICTIONS,
                prediction["id"],
                {
                    "deleted": True,
                    "deleted_at": datetime.now(timezone.utc)
                }
            )
            cleanup_results["old_predictions"] += 1
        
        # Clear old cache entries
        cache = get_cache()
        await cache.clear_pattern("temp:*")
        await cache.clear_pattern("session:expired:*")
        
        logger.info(
            "Data cleanup completed",
            expired_alerts=cleanup_results["expired_alerts"],
            old_notifications=cleanup_results["old_notifications"],
            old_predictions=cleanup_results["old_predictions"]
        )
        
        return cleanup_results
        
    except Exception as e:
        logger.error("Data cleanup failed", error=str(e))
        raise


@celery_app.task(bind=True)
@task_monitor.track_task("backup_data")
def backup_data(self, backup_type: str = "daily"):
    """
    Create data backup
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_backup_data_async(backup_type))
        loop.close()
        return result
        
    except Exception as e:
        logger.error("Data backup task failed", backup_type=backup_type, error=str(e))
        raise


async def _backup_data_async(backup_type: str):
    """Create backup of critical data"""
    try:
        from app.core.utils import generate_id
        
        backup_id = generate_id(f"BACKUP_{backup_type.upper()}_")
        backup_time = datetime.now(timezone.utc)
        
        # Determine backup scope based on type
        if backup_type == "daily":
            cutoff_time = backup_time - timedelta(days=1)
        elif backup_type == "weekly":
            cutoff_time = backup_time - timedelta(days=7)
        else:  # full
            cutoff_time = datetime.min.replace(tzinfo=timezone.utc)
        
        backup_data = {
            "id": backup_id,
            "type": backup_type,
            "created_at": backup_time,
            "data_counts": {}
        }
        
        # Backup critical collections
        collections_to_backup = [
            FirestoreCollections.USERS,
            FirestoreCollections.HEALTH_DATA,
            FirestoreCollections.WATER_QUALITY,
            FirestoreCollections.ALERTS,
            FirestoreCollections.ML_PREDICTIONS
        ]
        
        for collection in collections_to_backup:
            try:
                if backup_type == "full":
                    data = await firestore_service.query_collection(collection, limit=10000)
                else:
                    # Use appropriate timestamp field for each collection
                    timestamp_field = {
                        FirestoreCollections.USERS: "created_at",
                        FirestoreCollections.HEALTH_DATA: "report_time",
                        FirestoreCollections.WATER_QUALITY: "collection_time",
                        FirestoreCollections.ALERTS: "created_at",
                        FirestoreCollections.ML_PREDICTIONS: "prediction_timestamp"
                    }.get(collection, "created_at")
                    
                    data = await firestore_service.query_collection(
                        collection,
                        filters=[(timestamp_field, ">=", cutoff_time)],
                        limit=5000
                    )
                
                backup_data["data_counts"][collection] = len(data)
                
                # Store backup data (in production, send to cloud storage)
                backup_collection = f"backups_{collection}"
                for record in data:
                    record["backup_id"] = backup_id
                    record["backed_up_at"] = backup_time
                    await firestore_service.create_document(
                        backup_collection,
                        record,
                        f"{backup_id}_{record.get('id', generate_id())}"
                    )
                
            except Exception as e:
                logger.error(f"Failed to backup collection {collection}", error=str(e))
                backup_data["data_counts"][collection] = -1  # Error marker
        
        # Store backup metadata
        await firestore_service.create_document(
            "backup_metadata",
            backup_data,
            backup_id
        )
        
        logger.info(
            "Data backup completed",
            backup_id=backup_id,
            backup_type=backup_type,
            total_records=sum(count for count in backup_data["data_counts"].values() if count > 0)
        )
        
        return {
            "backup_id": backup_id,
            "backup_type": backup_type,
            "data_counts": backup_data["data_counts"],
            "created_at": backup_time.isoformat()
        }
        
    except Exception as e:
        logger.error("Data backup failed", backup_type=backup_type, error=str(e))
        raise


@celery_app.task(bind=True)
@task_monitor.track_task("process_water_quality_analysis")
def process_water_quality_analysis(self, analysis_request: Dict[str, Any]):
    """
    Background task to analyze water quality data and generate insights
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_process_water_quality_analysis_async(analysis_request))
        loop.close()
        return result
        
    except Exception as e:
        logger.error("Water quality analysis failed", error=str(e))
        raise


async def _process_water_quality_analysis_async(analysis_request: Dict[str, Any]):
    """Analyze water quality trends and patterns"""
    try:
        from app.core.utils import generate_id, calculate_water_quality_index
        
        region = analysis_request.get("region", {})
        time_period = analysis_request.get("time_period_days", 30)
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=time_period)
        
        # Get water quality data for the region
        filters = [("collection_time", ">=", cutoff_time)]
        if region.get("state"):
            filters.append(("state", "==", region["state"]))
        if region.get("district"):
            filters.append(("district", "==", region["district"]))
        
        water_quality_data = await firestore_service.query_collection(
            FirestoreCollections.WATER_QUALITY,
            filters=filters,
            limit=1000
        )
        
        if not water_quality_data:
            return {"status": "no_data", "message": "No water quality data found for analysis"}
        
        # Analyze trends
        analysis_results = {
            "id": generate_id("WQ_ANALYSIS_"),
            "region": region,
            "analysis_period": {
                "start_date": cutoff_time.isoformat(),
                "end_date": datetime.now(timezone.utc).isoformat(),
                "days": time_period
            },
            "data_points": len(water_quality_data),
            "summary_statistics": {},
            "trends": {},
            "quality_issues": [],
            "recommendations": [],
            "analyzed_at": datetime.now(timezone.utc)
        }
        
        # Calculate summary statistics
        ph_values = [d.get("ph_level", 7.0) for d in water_quality_data if d.get("ph_level")]
        turbidity_values = [d.get("turbidity", 0) for d in water_quality_data if d.get("turbidity")]
        
        if ph_values:
            analysis_results["summary_statistics"]["ph"] = {
                "average": sum(ph_values) / len(ph_values),
                "min": min(ph_values),
                "max": max(ph_values),
                "out_of_range_count": len([ph for ph in ph_values if ph < 6.5 or ph > 8.5])
            }
        
        if turbidity_values:
            analysis_results["summary_statistics"]["turbidity"] = {
                "average": sum(turbidity_values) / len(turbidity_values),
                "min": min(turbidity_values),
                "max": max(turbidity_values),
                "high_turbidity_count": len([t for t in turbidity_values if t > 5.0])
            }
        
        # Identify quality issues
        contaminated_sources = [d for d in water_quality_data if d.get("quality_status") == "requires_attention"]
        bacterial_contamination = [d for d in water_quality_data if d.get("bacterial_contamination") == True]
        
        if contaminated_sources:
            analysis_results["quality_issues"].append({
                "type": "contaminated_sources",
                "count": len(contaminated_sources),
                "percentage": (len(contaminated_sources) / len(water_quality_data)) * 100
            })
        
        if bacterial_contamination:
            analysis_results["quality_issues"].append({
                "type": "bacterial_contamination",
                "count": len(bacterial_contamination),
                "percentage": (len(bacterial_contamination) / len(water_quality_data)) * 100
            })
        
        # Generate recommendations
        if analysis_results["summary_statistics"].get("ph", {}).get("out_of_range_count", 0) > 0:
            analysis_results["recommendations"].append(
                "pH levels outside safe range detected. Check water treatment systems."
            )
        
        if analysis_results["summary_statistics"].get("turbidity", {}).get("high_turbidity_count", 0) > 0:
            analysis_results["recommendations"].append(
                "High turbidity levels detected. Implement filtration measures."
            )
        
        if len(contaminated_sources) > len(water_quality_data) * 0.1:  # More than 10% contaminated
            analysis_results["recommendations"].append(
                "Multiple contaminated water sources detected. Conduct comprehensive water system audit."
            )
        
        # Store analysis results
        await firestore_service.create_document(
            "water_quality_analyses",
            analysis_results,
            analysis_results["id"]
        )
        
        logger.info(
            "Water quality analysis completed",
            analysis_id=analysis_results["id"],
            data_points=len(water_quality_data),
            quality_issues=len(analysis_results["quality_issues"])
        )
        
        return analysis_results
        
    except Exception as e:
        logger.error("Water quality analysis processing failed", error=str(e))
        raise


@celery_app.task(bind=True)
@task_monitor.track_task("send_weekly_summary")
def send_weekly_summary(self):
    """
    Generate and send weekly health summary to authorities
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_weekly_summary_async())
        loop.close()
        return result
        
    except Exception as e:
        logger.error("Weekly summary task failed", error=str(e))
        raise


async def _send_weekly_summary_async():
    """Generate and distribute weekly health summary"""
    try:
        from app.core.utils import generate_id
        
        # Calculate week period
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        # Get authorities to notify
        authorities = await firestore_service.query_collection(
            FirestoreCollections.USERS,
            filters=[
                ("role", "in", ["admin", "state_health_authority", "local_authority"]),
                ("active", "==", True)
            ]
        )
        
        # Collect weekly data
        week_filters = [
            ("report_time", ">=", start_date) if "report_time" in ["report_time"] else 
            ("created_at", ">=", start_date),
            ("report_time", "<", end_date) if "report_time" in ["report_time"] else 
            ("created_at", "<", end_date)
        ]
        
        health_data = await firestore_service.query_collection(
            FirestoreCollections.HEALTH_DATA,
            filters=[("report_time", ">=", start_date), ("report_time", "<", end_date)]
        )
        
        water_quality_data = await firestore_service.query_collection(
            FirestoreCollections.WATER_QUALITY,
            filters=[("collection_time", ">=", start_date), ("collection_time", "<", end_date)]
        )
        
        alerts_data = await firestore_service.query_collection(
            FirestoreCollections.ALERTS,
            filters=[("created_at", ">=", start_date), ("created_at", "<", end_date)]
        )
        
        # Generate summary statistics
        summary_stats = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": 7
            },
            "health_reports": len(health_data),
            "water_quality_tests": len(water_quality_data),
            "alerts_issued": len(alerts_data),
            "critical_alerts": len([a for a in alerts_data if a.get("severity") == "critical"]),
            "high_risk_areas": len(set([h.get("district") for h in health_data if h.get("symptom_severity") == "severe"]))
        }
        
        # State-wise breakdown
        state_summary = {}
        for record in health_data:
            state = record.get("state", "unknown")
            if state not in state_summary:
                state_summary[state] = {"reports": 0, "severe_cases": 0}
            state_summary[state]["reports"] += 1
            if record.get("symptom_severity") == "severe":
                state_summary[state]["severe_cases"] += 1
        
        # Top symptoms
        symptom_counts = {}
        for record in health_data:
            for symptom in record.get("symptoms", []):
                symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
        
        top_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create weekly summary
        weekly_summary = {
            "id": generate_id("WEEKLY_SUMMARY_"),
            "type": "weekly_health_summary",
            "generated_at": datetime.now(timezone.utc),
            "summary_statistics": summary_stats,
            "geographic_breakdown": state_summary,
            "top_symptoms": top_symptoms,
            "key_insights": _generate_weekly_insights(summary_stats, top_symptoms, state_summary),
            "action_items": _generate_action_items(summary_stats, alerts_data)
        }
        
        # Store summary
        await firestore_service.create_document(
            "weekly_summaries",
            weekly_summary,
            weekly_summary["id"]
        )
        
        # Send to authorities
        notification_count = 0
        for authority in authorities:
            try:
                notification_data = {
                    "id": generate_id("WEEKLY_NOTIF_"),
                    "user_uid": authority["uid"],
                    "type": "weekly_summary",
                    "title": "Weekly Health Monitoring Summary",
                    "summary_id": weekly_summary["id"],
                    "created_at": datetime.now(timezone.utc),
                    "read": False
                }
                
                await firestore_service.create_document(
                    "notifications",
                    notification_data,
                    notification_data["id"]
                )
                
                # Send email notification for weekly summary
                if authority.get("email_enabled", True):
                    await send_weekly_email_summary(authority, weekly_summary)
                
                notification_count += 1
                
            except Exception as e:
                logger.error(
                    "Failed to send weekly summary to authority",
                    authority_uid=authority.get("uid"),
                    error=str(e)
                )
        
        logger.info(
            "Weekly summary generated and distributed",
            summary_id=weekly_summary["id"],
            authorities_notified=notification_count,
            health_reports=summary_stats["health_reports"]
        )
        
        return {
            "summary_id": weekly_summary["id"],
            "authorities_notified": notification_count,
            "summary_statistics": summary_stats
        }
        
    except Exception as e:
        logger.error("Weekly summary generation failed", error=str(e))
        raise


def _generate_weekly_insights(stats: Dict, symptoms: List, states: Dict) -> List[str]:
    """Generate key insights from weekly data"""
    insights = []
    
    if stats["critical_alerts"] > 0:
        insights.append(f"{stats['critical_alerts']} critical health alerts were issued this week.")
    
    if symptoms:
        top_symptom = symptoms[0]
        insights.append(f"Most reported symptom: {top_symptom[0]} ({top_symptom[1]} cases).")
    
    if stats["high_risk_areas"] > 0:
        insights.append(f"{stats['high_risk_areas']} districts reported severe health cases.")
    
    # Compare with previous week would go here in full implementation
    if stats["health_reports"] > 50:  # Threshold for concern
        insights.append("Above-average health reporting activity detected.")
    
    return insights


def _generate_action_items(stats: Dict, alerts: List[Dict]) -> List[str]:
    """Generate action items based on weekly data"""
    actions = []
    
    if stats["critical_alerts"] > 0:
        actions.append("Review and follow up on critical health alerts.")
    
    emergency_alerts = [a for a in alerts if a.get("alert_type") == "general" and a.get("severity") == "critical"]
    if emergency_alerts:
        actions.append("Investigate emergency health situations reported this week.")
    
    if stats["health_reports"] > 100:  # High activity threshold
        actions.append("Consider deploying additional health monitoring resources.")
    
    return actions


async def send_weekly_email_summary(authority: Dict[str, Any], summary: Dict[str, Any]) -> bool:
    """Send weekly summary via email"""
    try:
        # Mock email implementation
        logger.info(
            "Weekly email summary sent",
            authority_uid=authority["uid"],
            email=authority.get("email", "")[:10] + "...",
            summary_id=summary["id"]
        )
        
        await asyncio.sleep(0.1)  # Simulate email sending delay
        return True
        
    except Exception as e:
        logger.error("Weekly email summary failed", authority_uid=authority.get("uid"), error=str(e))
        return False


# Periodic tasks schedule
celery_app.conf.beat_schedule = {
    # Run daily report generation every day at 1:00 AM UTC
    'generate-daily-reports': {
        'task': 'app.tasks.generate_daily_report',
        'schedule': crontab(hour=1, minute=0),
    },
    
    # Clean up old data every day at 2:00 AM UTC
    'cleanup-old-data': {
        'task': 'app.tasks.cleanup_old_data',
        'schedule': crontab(hour=2, minute=0),
    },
    
    # Weekly summary every Monday at 8:00 AM UTC
    'send-weekly-summary': {
        'task': 'app.tasks.send_weekly_summary',
        'schedule': crontab(hour=8, minute=0, day_of_week=1),
    },
    
    # Daily backup every day at 3:00 AM UTC
    'daily-backup': {
        'task': 'app.tasks.backup_data',
        'schedule': crontab(hour=3, minute=0),
        'args': ('daily',)
    },
    
    # Weekly backup every Sunday at 4:00 AM UTC
    'weekly-backup': {
        'task': 'app.tasks.backup_data',
        'schedule': crontab(hour=4, minute=0, day_of_week=0),
        'args': ('weekly',)
    },
}

# Set timezone for beat schedule
celery_app.conf.timezone = 'UTC'


# Utility functions for task management
def queue_alert_notification(alert_id: str, alert_data: Dict[str, Any], creator_uid: str):
    """Queue alert notification task"""
    return send_alert_notifications.delay(alert_id, alert_data, creator_uid)


def queue_ml_prediction(prediction_data: Dict[str, Any]):
    """Queue ML prediction task"""
    return process_ml_prediction.delay(prediction_data)


def queue_water_quality_analysis(analysis_request: Dict[str, Any]):
    """Queue water quality analysis task"""
    return process_water_quality_analysis.delay(analysis_request)


# Task status checking
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a background task"""
    try:
        task_result = celery_app.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result if task_result.ready() else None,
            "info": task_result.info,
            "successful": task_result.successful(),
            "failed": task_result.failed(),
            "ready": task_result.ready()
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "ERROR",
            "error": str(e)
        }


# Health check for Celery workers
@celery_app.task
def health_check():
    """Health check task for monitoring Celery workers"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker": "celery_worker"
    }


# Export task management functions
__all__ = [
    "celery_app",
    "send_alert_notifications",
    "process_ml_prediction",
    "generate_daily_report",
    "cleanup_old_data",
    "backup_data",
    "process_water_quality_analysis",
    "send_weekly_summary",
    "queue_alert_notification",
    "queue_ml_prediction",
    "queue_water_quality_analysis",
    "get_task_status",
    "health_check"
]
            