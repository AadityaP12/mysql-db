import firebase_admin
from firebase_admin import credentials, firestore # FIX: Added 'credentials' for completeness
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional
import logging

from app.config import settings

# Firestore client (Primary database)
_firestore_client = None

# PostgreSQL setup (Optional for structured data)
engine = None
SessionLocal = None
Base = declarative_base()

logger = logging.getLogger(__name__)


def init_firebase():
    """
    Initialize the Firebase Admin SDK if it hasn't been already.
    """
    if not firebase_admin._apps:
        try:
            if settings.FIREBASE_SERVICE_ACCOUNT_PATH:
                cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT_PATH)
                firebase_admin.initialize_app(cred, {
                    'projectId': settings.FIREBASE_PROJECT_ID
                })
                logger.info("Firebase initialized with service account JSON.")
            else:
                raise ValueError("FIREBASE_SERVICE_ACCOUNT_PATH not set in .env")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
            raise


def get_firestore_client():
    """
    Get Firestore database client (singleton pattern)
    """
    global _firestore_client
    
    if _firestore_client is None:
        try:
            # The client is created after ensuring the app is initialized.
            _firestore_client = firestore.client()
            logger.info("Firestore client retrieved successfully")
        except Exception as e:
            logger.error(f"Failed to retrieve Firestore client: {e}")
            raise e
    
    return _firestore_client


def init_postgres():
    """
    Initialize PostgreSQL database (optional)
    """
    global engine, SessionLocal
    
    if settings.DATABASE_URL:
        try:
            engine = create_engine(
                settings.DATABASE_URL,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            logger.info("PostgreSQL database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            engine = None
            SessionLocal = None


def get_postgres_session():
    """
    Get PostgreSQL database session
    """
    if SessionLocal is None:
        raise Exception("PostgreSQL not initialized")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Firestore Collections
class FirestoreCollections:
    USERS = "users"
    HEALTH_DATA = "health_data" 
    WATER_QUALITY = "water_quality"
    ALERTS = "alerts"
    REGIONS = "regions"
    HOUSEHOLDS = "households"
    DISEASE_REPORTS = "disease_reports"
    IOT_SENSORS = "iot_sensors"
    ML_PREDICTIONS = "ml_predictions"
    AUDIT_LOGS = "audit_logs"


class FirestoreService:
    """
    Firestore database service class
    """
    
    def __init__(self):
        self.db = get_firestore_client()
    
    async def create_document(self, collection: str, data: dict, doc_id: str = None) -> str:
        """
        Create a new document in Firestore
        """
        try:
            if doc_id:
                doc_ref = self.db.collection(collection).document(doc_id)
                doc_ref.set(data)
                return doc_id
            else:
                doc_ref = self.db.collection(collection).add(data)[1]
                return doc_ref.id
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            raise
    
    async def get_document(self, collection: str, doc_id: str) -> Optional[dict]:
        """
        Get a document by ID
        """
        try:
            doc_ref = self.db.collection(collection).document(doc_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            raise
    
    async def update_document(self, collection: str, doc_id: str, data: dict) -> bool:
        """
        Update a document
        """
        try:
            doc_ref = self.db.collection(collection).document(doc_id)
            doc_ref.update(data)
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise
    
    async def delete_document(self, collection: str, doc_id: str) -> bool:
        """
        Delete a document
        """
        try:
            doc_ref = self.db.collection(collection).document(doc_id)
            doc_ref.delete()
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    async def query_collection(
        self, 
        collection: str, 
        filters: list = None, 
        order_by: str = None, 
        limit: int = None
    ) -> list:
        """
        Query a collection with filters
        filters: list of tuples (field, operator, value)
        """
        try:
            query = self.db.collection(collection)
            
            # Apply filters
            if filters:
                for field, operator, value in filters:
                    query = query.where(field, operator, value)
            
            # Apply ordering
            if order_by:
                query = query.order_by(order_by)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Execute query
            docs = query.stream()
            results = []
            
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                results.append(data)
            
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise
    
    async def batch_write(self, operations: list) -> bool:
        """
        Perform batch write operations
        operations: list of dicts with keys: 'action', 'collection', 'doc_id', 'data'
        """
        try:
            batch = self.db.batch()
            
            for op in operations:
                doc_ref = self.db.collection(op['collection']).document(op['doc_id'])
                
                if op['action'] == 'create' or op['action'] == 'set':
                    batch.set(doc_ref, op['data'])
                elif op['action'] == 'update':
                    batch.update(doc_ref, op['data'])
                elif op['action'] == 'delete':
                    batch.delete(doc_ref)
            
            batch.commit()
            return True
        except Exception as e:
            logger.error(f"Error in batch write: {e}")
            raise


# FIX: Initialize Firebase before creating any services that depend on it.
init_firebase()

# Initialize services
firestore_service = FirestoreService()

# Initialize PostgreSQL if configured
if settings.DATABASE_URL:
    init_postgres()