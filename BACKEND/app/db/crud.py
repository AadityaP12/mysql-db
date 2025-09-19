from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from app.db.database import firestore_service, FirestoreCollections
from app.core.utils import generate_id, calculate_distance, validate_coordinates
from app.schemas.data import WaterQualityData, HealthData, HouseholdData, SensorData
from app.schemas.auth import UserRegistration, UserProfile

logger = logging.getLogger(__name__)


class CRUDBase:
    """
    Base CRUD operations class for Firestore database operations
    """
    
    def __init__(self, collection: str):
        self.collection = collection
        self.db = firestore_service

    async def create(self, data: Dict[str, Any], doc_id: str = None) -> str:
        """Create a new document"""
        try:
            document_id = await self.db.create_document(
                self.collection, 
                data, 
                doc_id or generate_id()
            )
            logger.info(f"Created document {document_id} in {self.collection}")
            return document_id
        except Exception as e:
            logger.error(f"Error creating document in {self.collection}: {str(e)}")
            raise

    async def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            return await self.db.get_document(self.collection, doc_id)
        except Exception as e:
            logger.error(f"Error getting document {doc_id} from {self.collection}: {str(e)}")
            raise

    async def update(self, doc_id: str, data: Dict[str, Any]) -> bool:
        """Update a document"""
        try:
            data['updated_at'] = datetime.now(timezone.utc)
            return await self.db.update_document(self.collection, doc_id, data)
        except Exception as e:
            logger.error(f"Error updating document {doc_id} in {self.collection}: {str(e)}")
            raise

    async def delete(self, doc_id: str) -> bool:
        """Soft delete a document"""
        try:
            return await self.db.update_document(
                self.collection, 
                doc_id, 
                {
                    'deleted': True,
                    'deleted_at': datetime.now(timezone.utc)
                }
            )
        except Exception as e:
            logger.error(f"Error deleting document {doc_id} from {self.collection}: {str(e)}")
            raise

    async def query(
        self, 
        filters: List[tuple] = None, 
        order_by: str = None, 
        limit: int = None,
        exclude_deleted: bool = True
    ) -> List[Dict[str, Any]]:
        """Query documents with filters"""
        try:
            # Add deleted filter if needed
            if exclude_deleted:
                if not filters:
                    filters = []
                filters.append(('deleted', '==', False))
            
            return await self.db.query_collection(
                self.collection,
                filters=filters,
                order_by=order_by,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error querying {self.collection}: {str(e)}")
            raise

    async def count(self, filters: List[tuple] = None) -> int:
        """Count documents matching filters"""
        try:
            results = await self.query(filters=filters)
            return len(results)
        except Exception as e:
            logger.error(f"Error counting documents in {self.collection}: {str(e)}")
            raise


class UserCRUD(CRUDBase):
    """CRUD operations for users"""
    
    def __init__(self):
        super().__init__(FirestoreCollections.USERS)
    
    async def create_user(self, user_data: UserRegistration, uid: str) -> Dict[str, Any]:
        """Create a new user profile"""
        try:
            user_profile = {
                "uid": uid,
                "email": user_data.email,
                "full_name": user_data.full_name,
                "phone_number": user_data.phone_number,
                "role": user_data.role.value,
                "region": user_data.region,
                "state": user_data.state,
                "district": user_data.district,
                "block": user_data.block,
                "village": user_data.village,
                "organization": user_data.organization,
                "employee_id": user_data.employee_id,
                "preferred_language": user_data.preferred_language,
                "verified": False,
                "active": True,
                "created_at": datetime.now(timezone.utc),
                "last_login": None,
                "profile_completed": True,
                "deleted": False
            }
            
            await self.create(user_profile, uid)
            return user_profile
            
        except Exception as e:
            logger.error(f"Error creating user profile: {str(e)}")
            raise

    async def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            users = await self.query(filters=[('email', '==', email)])
            return users[0] if users else None
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {str(e)}")
            raise

    async def get_by_role(self, role: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get users by role"""
        try:
            return await self.query(
                filters=[('role', '==', role)],
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error getting users by role {role}: {str(e)}")
            raise

    async def update_last_login(self, uid: str) -> bool:
        """Update user's last login timestamp"""
        try:
            return await self.update(uid, {'last_login': datetime.now(timezone.utc)})
        except Exception as e:
            logger.error(f"Error updating last login for user {uid}: {str(e)}")
            raise

    async def get_users_by_location(
        self, 
        state: str = None, 
        district: str = None, 
        block: str = None
    ) -> List[Dict[str, Any]]:
        """Get users by location"""
        try:
            filters = []
            if state:
                filters.append(('state', '==', state))
            if district:
                filters.append(('district', '==', district))
            if block:
                filters.append(('block', '==', block))
            
            return await self.query(filters=filters)
        except Exception as e:
            logger.error(f"Error getting users by location: {str(e)}")
            raise


class WaterQualityCRUD(CRUDBase):
    """CRUD operations for water quality data"""
    
    def __init__(self):
        super().__init__(FirestoreCollections.WATER_QUALITY)
    
    async def create_water_quality_record(
        self, 
        water_data: WaterQualityData, 
        uploaded_by: str
    ) -> str:
        """Create a new water quality record"""
        try:
            record = water_data.dict()
            record.update({
                "id": generate_id("WQ_"),
                "uploaded_by": uploaded_by,
                "upload_timestamp": datetime.now(timezone.utc),
                "processed": False,
                "quality_status": self._assess_water_quality(water_data),
                "deleted": False
            })
            
            return await self.create(record, record["id"])
            
        except Exception as e:
            logger.error(f"Error creating water quality record: {str(e)}")
            raise

    async def get_by_location_and_time(
        self,
        location: Dict[str, float],
        radius_km: float = 5.0,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """Get water quality records by location and time"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            # Get all recent records first
            recent_records = await self.query(
                filters=[('collection_time', '>=', start_time)],
                order_by='collection_time'
            )
            
            # Filter by location (Firestore doesn't support geo queries directly)
            nearby_records = []
            for record in recent_records:
                record_location = record.get('location', {})
                if 'latitude' in record_location and 'longitude' in record_location:
                    distance = calculate_distance(
                        location['latitude'], location['longitude'],
                        record_location['latitude'], record_location['longitude']
                    )
                    if distance <= radius_km:
                        record['distance_km'] = distance
                        nearby_records.append(record)
            
            return sorted(nearby_records, key=lambda x: x.get('distance_km', 0))
            
        except Exception as e:
            logger.error(f"Error getting water quality by location: {str(e)}")
            raise

    async def get_contaminated_sources(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get contaminated water sources in the last N days"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            return await self.query(
                filters=[
                    ('collection_time', '>=', start_time),
                    ('quality_status', '==', 'requires_attention')
                ],
                order_by='collection_time'
            )
        except Exception as e:
            logger.error(f"Error getting contaminated sources: {str(e)}")
            raise

    async def get_by_sensor(self, sensor_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get water quality records by sensor ID"""
        try:
            return await self.query(
                filters=[('sensor_id', '==', sensor_id)],
                order_by='collection_time',
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error getting records by sensor {sensor_id}: {str(e)}")
            raise

    def _assess_water_quality(self, water_data: WaterQualityData) -> str:
        """Assess water quality status based on parameters"""
        issues = 0
        
        # pH assessment
        if water_data.ph_level < 6.5 or water_data.ph_level > 8.5:
            issues += 1
        
        # Turbidity assessment
        if water_data.turbidity > 5.0:
            issues += 2  # High turbidity is more serious
        
        # Chlorine assessment
        if water_data.residual_chlorine is not None and water_data.residual_chlorine < 0.2:
            issues += 1
        
        # Bacterial contamination
        if water_data.bacterial_contamination:
            issues += 3  # Most serious issue
        
        if issues >= 3:
            return "requires_attention"
        elif issues >= 1:
            return "caution_required"
        else:
            return "acceptable"


class HealthDataCRUD(CRUDBase):
    """CRUD operations for health data"""
    
    def __init__(self):
        super().__init__(FirestoreCollections.HEALTH_DATA)
    
    async def create_health_record(
        self, 
        health_data: HealthData, 
        reported_by: str
    ) -> str:
        """Create a new health record"""
        try:
            record = health_data.dict()
            record.update({
                "id": generate_id("HD_"),
                "reported_by": reported_by,
                "upload_timestamp": datetime.now(timezone.utc),
                "processed": False,
                "risk_assessed": False,
                "severity_score": self._calculate_severity_score(health_data),
                "deleted": False
            })
            
            return await self.create(record, record["id"])
            
        except Exception as e:
            logger.error(f"Error creating health record: {str(e)}")
            raise

    async def get_symptoms_by_location_and_time(
        self,
        location: Dict[str, float],
        radius_km: float = 10.0,
        days_back: int = 14
    ) -> List[Dict[str, Any]]:
        """Get symptom reports by location and time"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Get recent health records
            recent_records = await self.query(
                filters=[('report_time', '>=', start_time)],
                order_by='report_time'
            )
            
            # Filter by location
            nearby_records = []
            for record in recent_records:
                record_location = record.get('location', {})
                if 'latitude' in record_location and 'longitude' in record_location:
                    distance = calculate_distance(
                        location['latitude'], location['longitude'],
                        record_location['latitude'], record_location['longitude']
                    )
                    if distance <= radius_km:
                        record['distance_km'] = distance
                        nearby_records.append(record)
            
            return nearby_records
            
        except Exception as e:
            logger.error(f"Error getting symptoms by location: {str(e)}")
            raise

    async def get_symptom_statistics(
        self, 
        days_back: int = 30,
        location_filters: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Get symptom statistics"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            filters = [('report_time', '>=', start_time)]
            if location_filters:
                for key, value in location_filters.items():
                    filters.append((key, '==', value))
            
            records = await self.query(filters=filters)
            
            # Count symptoms
            symptom_counts = {}
            severity_counts = {'mild': 0, 'moderate': 0, 'severe': 0}
            age_groups = {'0-18': 0, '19-60': 0, '60+': 0}
            
            for record in records:
                # Count symptoms
                for symptom in record.get('symptoms', []):
                    symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
                
                # Count severity
                severity = record.get('symptom_severity', 'mild')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # Count age groups
                age = record.get('age', 0)
                if age <= 18:
                    age_groups['0-18'] += 1
                elif age <= 60:
                    age_groups['19-60'] += 1
                else:
                    age_groups['60+'] += 1
            
            return {
                'total_reports': len(records),
                'symptom_counts': symptom_counts,
                'severity_distribution': severity_counts,
                'age_distribution': age_groups,
                'time_period_days': days_back
            }
            
        except Exception as e:
            logger.error(f"Error getting symptom statistics: {str(e)}")
            raise

    async def get_high_risk_cases(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get high-risk health cases"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            high_risk_records = await self.query(
                filters=[
                    ('report_time', '>=', start_time),
                    ('severity_score', '>=', 7)  # High severity threshold
                ],
                order_by='report_time'
            )
            
            return high_risk_records
            
        except Exception as e:
            logger.error(f"Error getting high-risk cases: {str(e)}")
            raise

    def _calculate_severity_score(self, health_data: HealthData) -> int:
        """Calculate severity score (1-10) based on symptoms and conditions"""
        score = 1
        
        # Base severity from enum
        severity_map = {'mild': 2, 'moderate': 5, 'severe': 8}
        score = severity_map.get(health_data.symptom_severity.value, 2)
        
        # High-risk symptoms
        high_risk_symptoms = [
            'severe_diarrhea', 'bloody_stool', 'severe_dehydration',
            'high_fever', 'persistent_vomiting', 'severe_abdominal_pain'
        ]
        
        for symptom in health_data.symptoms:
            if symptom in high_risk_symptoms:
                score += 2
        
        # Age factors (children and elderly at higher risk)
        if health_data.age < 5 or health_data.age > 65:
            score += 1
        
        # Fever temperature
        if health_data.fever_temperature and health_data.fever_temperature > 39.0:
            score += 1
        
        # Hospitalization required
        if health_data.hospitalization_required:
            score += 2
        
        return min(10, score)  # Cap at 10


class HouseholdCRUD(CRUDBase):
    """CRUD operations for household data"""
    
    def __init__(self):
        super().__init__(FirestoreCollections.HOUSEHOLDS)
    
    async def create_household(
        self, 
        household_data: HouseholdData, 
        registered_by: str
    ) -> str:
        """Create a new household record"""
        try:
            record = household_data.dict()
            record.update({
                "registered_by": registered_by,
                "registration_timestamp": datetime.now(timezone.utc),
                "active": True,
                "deleted": False
            })
            
            return await self.create(record, household_data.house_id)
            
        except Exception as e:
            logger.error(f"Error creating household record: {str(e)}")
            raise

    async def get_by_water_source(self, water_source: str) -> List[Dict[str, Any]]:
        """Get households by water source type"""
        try:
            return await self.query(filters=[('water_source', '==', water_source)])
        except Exception as e:
            logger.error(f"Error getting households by water source: {str(e)}")
            raise

    async def get_households_in_radius(
        self,
        center_location: Dict[str, float],
        radius_km: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Get households within radius of a location"""
        try:
            all_households = await self.query()
            
            nearby_households = []
            for household in all_households:
                hh_location = household.get('location', {})
                if 'latitude' in hh_location and 'longitude' in hh_location:
                    distance = calculate_distance(
                        center_location['latitude'], center_location['longitude'],
                        hh_location['latitude'], hh_location['longitude']
                    )
                    if distance <= radius_km:
                        household['distance_km'] = distance
                        nearby_households.append(household)
            
            return sorted(nearby_households, key=lambda x: x.get('distance_km', 0))
            
        except Exception as e:
            logger.error(f"Error getting households in radius: {str(e)}")
            raise

    async def get_population_statistics(
        self, 
        location_filters: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Get population statistics for households"""
        try:
            filters = []
            if location_filters:
                for key, value in location_filters.items():
                    filters.append((key, '==', value))
            
            households = await self.query(filters=filters)
            
            total_households = len(households)
            total_population = sum(hh.get('total_members', 0) for hh in households)
            
            # Water source distribution
            water_source_counts = {}
            sanitation_counts = {}
            
            for hh in households:
                # Water sources
                water_source = hh.get('water_source', 'unknown')
                water_source_counts[water_source] = water_source_counts.get(water_source, 0) + 1
                
                # Sanitation
                sanitation = hh.get('sanitation_type', 'unknown')
                sanitation_counts[sanitation] = sanitation_counts.get(sanitation, 0) + 1
            
            return {
                'total_households': total_households,
                'total_population': total_population,
                'average_household_size': total_population / total_households if total_households > 0 else 0,
                'water_source_distribution': water_source_counts,
                'sanitation_distribution': sanitation_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting population statistics: {str(e)}")
            raise


class AlertCRUD(CRUDBase):
    """CRUD operations for alerts"""
    
    def __init__(self):
        super().__init__(FirestoreCollections.ALERTS)
    
    async def create_alert(self, alert_data: Dict[str, Any], created_by: str) -> str:
        """Create a new alert"""
        try:
            alert_record = alert_data.copy()
            alert_record.update({
                "id": generate_id("ALERT_"),
                "created_by": created_by,
                "created_at": datetime.now(timezone.utc),
                "status": "active",
                "recipients_count": 0,
                "read_count": 0,
                "deleted": False
            })
            
            return await self.create(alert_record, alert_record["id"])
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            raise

    async def get_active_alerts(
        self, 
        location_filters: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """Get active alerts"""
        try:
            filters = [('status', '==', 'active')]
            if location_filters:
                for key, value in location_filters.items():
                    filters.append((key, '==', value))
            
            return await self.query(
                filters=filters,
                order_by='created_at'
            )
        except Exception as e:
            logger.error(f"Error getting active alerts: {str(e)}")
            raise

    async def get_alerts_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get alerts by severity level"""
        try:
            return await self.query(
                filters=[
                    ('severity', '==', severity),
                    ('status', '==', 'active')
                ],
                order_by='created_at'
            )
        except Exception as e:
            logger.error(f"Error getting alerts by severity: {str(e)}")
            raise

    async def expire_old_alerts(self, hours_old: int = 24) -> int:
        """Expire alerts older than specified hours"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_old)
            
            old_alerts = await self.query(
                filters=[
                    ('created_at', '<', cutoff_time),
                    ('status', '==', 'active')
                ]
            )
            
            expired_count = 0
            for alert in old_alerts:
                await self.update(alert['id'], {'status': 'expired'})
                expired_count += 1
            
            return expired_count
            
        except Exception as e:
            logger.error(f"Error expiring old alerts: {str(e)}")
            raise


class PredictionCRUD(CRUDBase):
    """CRUD operations for ML predictions"""
    
    def __init__(self):
        super().__init__(FirestoreCollections.ML_PREDICTIONS)
    
    async def create_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Create a new prediction record"""
        try:
            record = prediction_data.copy()
            record.update({
                "id": generate_id("PRED_"),
                "created_at": datetime.now(timezone.utc),
                "validated": False,
                "deleted": False
            })
            
            return await self.create(record, record["id"])
            
        except Exception as e:
            logger.error(f"Error creating prediction: {str(e)}")
            raise

    async def get_predictions_for_validation(self, days_old: int = 7) -> List[Dict[str, Any]]:
        """Get predictions ready for validation"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days_old)
            end_time = datetime.now(timezone.utc) - timedelta(days=1)  # At least 1 day old
            
            return await self.query(
                filters=[
                    ('created_at', '>=', start_time),
                    ('created_at', '<=', end_time),
                    ('validated', '==', False)
                ],
                order_by='created_at'
            )
        except Exception as e:
            logger.error(f"Error getting predictions for validation: {str(e)}")
            raise

    async def update_prediction_accuracy(
        self, 
        prediction_id: str, 
        actual_outcome: bool, 
        validated_by: str
    ) -> bool:
        """Update prediction with actual outcome"""
        try:
            prediction = await self.get_by_id(prediction_id)
            if not prediction:
                return False
            
            predicted_risk = prediction.get('risk_score', 0.5)
            predicted_outcome = predicted_risk > 0.5
            accuracy = 1.0 if predicted_outcome == actual_outcome else 0.0
            
            update_data = {
                'validated': True,
                'actual_outcome': actual_outcome,
                'accuracy_score': accuracy,
                'validated_by': validated_by,
                'validated_at': datetime.now(timezone.utc)
            }
            
            return await self.update(prediction_id, update_data)
            
        except Exception as e:
            logger.error(f"Error updating prediction accuracy: {str(e)}")
            raise


# Initialize CRUD instances
user_crud = UserCRUD()
water_quality_crud = WaterQualityCRUD()
health_data_crud = HealthDataCRUD()
household_crud = HouseholdCRUD()
alert_crud = AlertCRUD()
prediction_crud = PredictionCRUD()