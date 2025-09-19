from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from typing import List
import os
import json


class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "Water Disease Monitoring API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]

    # Firebase Configuration
    FIREBASE_SERVICE_ACCOUNT: dict = {}
    FIREBASE_PROJECT_ID: str = ""
    FIREBASE_SERVICE_ACCOUNT_PATH: str = ""

    # Database (PostgreSQL - Optional for structured data)
    DATABASE_URL: str = ""
    POSTGRES_DB: str = "water_disease_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # Redis Configuration for Caching and Session Management
    REDIS_URL: str = ""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_SSL: bool = False
    REDIS_EXPIRE_SECONDS: int = 3600  # 1 hour default

    # Celery Configuration for Background Tasks
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_ENABLE_UTC: bool = True

    # Rate Limiting Configuration
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REDIS_URL: str = ""
    DEFAULT_RATE_LIMIT: str = "100/minute"
    AUTH_RATE_LIMIT: str = "10/minute"
    UPLOAD_RATE_LIMIT: str = "50/minute"
    ML_RATE_LIMIT: str = "20/minute"

    # Caching Configuration
    CACHE_ENABLED: bool = True
    CACHE_DEFAULT_TTL: int = 300
    CACHE_USER_PROFILE_TTL: int = 1800
    CACHE_WATER_QUALITY_TTL: int = 600
    CACHE_PREDICTIONS_TTL: int = 3600
    CACHE_ALERTS_TTL: int = 120

    # ML Model Configuration
    MODEL_PATH: str = "app/ml/models/"
    MODEL_VERSION: str = "v1.0"
    RISK_THRESHOLD_LOW: float = 0.3
    RISK_THRESHOLD_HIGH: float = 0.7
    ML_MODEL_CACHE_TTL: int = 7200
    PREDICTION_BATCH_SIZE: int = 100

    # SMS/Alert Configuration
    SMS_API_KEY: str = ""
    SMS_SENDER_ID: str = "HEALTH"
    SMS_RATE_LIMIT: int = 1000

    # Security
    SECRET_KEY: str = "AIzaSyC6pNVgbv6KLkeKg19W_9co8FPztkbemrc"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    SESSION_EXPIRE_HOURS: int = 24

    # Monitoring and Observability
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8001
    LOG_LEVEL: str = "INFO"
    STRUCTURED_LOGGING: bool = True
    SENTRY_DSN: str = ""

    # Health Check Configuration
    HEALTH_CHECK_ENABLED: bool = True
    HEALTH_CHECK_REDIS_TIMEOUT: int = 5
    HEALTH_CHECK_DB_TIMEOUT: int = 10
    HEALTH_CHECK_FIREBASE_TIMEOUT: int = 15

    # Performance Configuration
    MAX_WORKERS: int = 4
    MAX_CONNECTIONS: int = 100
    CONNECTION_TIMEOUT: int = 30
    REQUEST_TIMEOUT: int = 60

    # API Versioning
    API_V1_STR: str = "/api/v1"
    API_VERSION: str = "1.0"

    # Backup and Data Retention
    DATA_RETENTION_DAYS: int = 365
    BACKUP_ENABLED: bool = True
    BACKUP_INTERVAL_HOURS: int = 6

    # Geographic Bounds for Northeast India
    GEO_BOUNDS: dict = {
        "lat_min": 21.0,
        "lat_max": 30.0,
        "lon_min": 87.0,
        "lon_max": 98.0
    }

    # Regional Language Support
    SUPPORTED_LANGUAGES: List[str] = ["en", "hi", "as", "bn", "mni"]
    DEFAULT_LANGUAGE: str = "en"

    # Feature Flags
    ENABLE_ML_PREDICTIONS: bool = True
    ENABLE_BACKGROUND_TASKS: bool = True
    ENABLE_SMS_ALERTS: bool = True
    ENABLE_PUSH_NOTIFICATIONS: bool = True
    ENABLE_DATA_EXPORT: bool = True
    ENABLE_BATCH_PROCESSING: bool = True

    # ✅ Pydantic v2 way of replacing `class Config`
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )

    # ✅ Use Pydantic validator instead of `__init__`
    @model_validator(mode="after")
    def build_urls(self) -> "Settings":
        # Build Postgres URL if not provided
        if not self.DATABASE_URL and all([
            self.POSTGRES_USER,
            self.POSTGRES_PASSWORD,
            self.POSTGRES_HOST,
            self.POSTGRES_DB
        ]):
            self.DATABASE_URL = (
                f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
                f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )

        # Build Redis URLs if missing
        if not self.REDIS_URL:
            auth_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
            protocol = "rediss" if self.REDIS_SSL else "redis"
            self.REDIS_URL = f"{protocol}://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

        if not self.CELERY_BROKER_URL:
            self.CELERY_BROKER_URL = f"{self.REDIS_URL}/1"

        if not self.CELERY_RESULT_BACKEND:
            self.CELERY_RESULT_BACKEND = f"{self.REDIS_URL}/2"

        if not self.RATE_LIMIT_REDIS_URL:
            self.RATE_LIMIT_REDIS_URL = f"{self.REDIS_URL}/3"

        # Load Firebase service account if path provided
        if self.FIREBASE_SERVICE_ACCOUNT_PATH and os.path.exists(self.FIREBASE_SERVICE_ACCOUNT_PATH):
            with open(self.FIREBASE_SERVICE_ACCOUNT_PATH, "r") as f:
                self.FIREBASE_SERVICE_ACCOUNT = json.load(f)

        return self

    # Redis config dict
    @property
    def redis_config(self) -> dict:
        return {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "db": self.REDIS_DB,
            "password": self.REDIS_PASSWORD or None,
            "ssl": self.REDIS_SSL,
            "decode_responses": True,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "retry_on_timeout": True,
            "health_check_interval": 30
        }

    # Celery config dict
    @property
    def celery_config(self) -> dict:
        return {
            "broker_url": self.CELERY_BROKER_URL,
            "result_backend": self.CELERY_RESULT_BACKEND,
            "task_serializer": self.CELERY_TASK_SERIALIZER,
            "result_serializer": self.CELERY_RESULT_SERIALIZER,
            "accept_content": self.CELERY_ACCEPT_CONTENT,
            "timezone": self.CELERY_TIMEZONE,
            "enable_utc": self.CELERY_ENABLE_UTC,
            "task_routes": {
                'app.tasks.send_alert_notifications': {'queue': 'alerts'},
                'app.tasks.process_ml_prediction': {'queue': 'ml'},
                'app.tasks.generate_reports': {'queue': 'reports'},
                'app.tasks.cleanup_old_data': {'queue': 'maintenance'},
            },
            "task_default_queue": "default",
            "worker_prefetch_multiplier": 1,
            "task_acks_late": True,
            "worker_disable_rate_limits": False,
            "task_ignore_result": False,
            "result_expires": 3600,
            "task_soft_time_limit": 300,
            "task_time_limit": 600,
        }


# Export global settings instance
settings = Settings()
