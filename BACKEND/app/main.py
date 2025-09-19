from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import firebase_admin
from firebase_admin import credentials
import asyncio
import time
import logging
from typing import Dict, Any
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.api import routes_auth, routes_data, routes_alerts, routes_ml
from app.core.utils import create_response
from app.core.cache import redis_client, get_cache, init_redis
from app.core.monitoring import setup_logging, setup_metrics
from app.core.health_checks import health_check_router
from app.db.database import firestore_service

# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)

# Setup metrics
setup_metrics()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", 
    "Total HTTP requests", 
    ["method", "endpoint", "status_code"]
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds"
)
ACTIVE_CONNECTIONS = Gauge(
    "http_active_connections", 
    "Number of active HTTP connections"
)
ML_PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Total ML predictions made",
    ["prediction_type", "risk_level"]
)
ALERTS_SENT_TOTAL = Counter(
    "alerts_sent_total",
    "Total alerts sent",
    ["alert_type", "severity"]
)
DATABASE_OPERATIONS = Counter(
    "database_operations_total",
    "Total database operations",
    ["operation", "collection", "status"]
)

# Rate limiter setup
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.RATE_LIMIT_REDIS_URL,
    default_limits=[settings.DEFAULT_RATE_LIMIT] if settings.RATE_LIMIT_ENABLED else []
)

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager for startup and shutdown events"""
    logger.info("Starting Water-Borne Disease Monitoring API")

    # --- Firebase init ---
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT)
            firebase_admin.initialize_app(cred)
        logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.error("Firebase initialization failed", error=str(e))
        raise

    # --- Redis init ---
    try:
        redis = await init_redis()  # ensures redis_client is created and pinged
        if redis:
            logger.info("Redis connection established successfully")
        else:
            logger.warning("Redis disabled or not available")
    except Exception as e:
        logger.error("Redis initialization failed", error=str(e))
        if settings.CACHE_ENABLED:
            logger.warning("Continuing without cache functionality")

    # --- Firestore init ---
    try:
        await firestore_service.query_collection("users", limit=1)
        logger.info("Firestore connection verified")
    except Exception as e:
        logger.error("Firestore connection failed", error=str(e))
        raise

    # --- ML model init ---
    try:
        # TODO: load your ML models here
        logger.info("ML models initialized")
    except Exception as e:
        logger.warning("ML model initialization failed", error=str(e))

    logger.info("Application startup completed successfully")

    # --- Yield control to FastAPI ---
    yield

    # --- Cleanup on shutdown ---
    logger.info("Shutting down application")

    try:
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")
    except Exception as e:
        logger.error("Error closing Redis connection", error=str(e))

    logger.info("Application shutdown completed")


app = FastAPI(
    title="Water-Borne Disease Monitoring API",
    description="Smart Community Health Monitoring and Early Warning System for Northeast India",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# Add rate limiting to app
if settings.RATE_LIMIT_ENABLED:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS if settings.ALLOWED_HOSTS != ["*"] else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics for monitoring"""
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        # Add custom headers
        response.headers["X-Response-Time"] = str(duration)
        response.headers["X-API-Version"] = settings.API_VERSION
        
        return response
    
    except Exception as e:
        # Record error metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=500
        ).inc()
        
        logger.error(
            "Request processing failed",
            method=request.method,
            path=request.url.path,
            error=str(e)
        )
        raise
    
    finally:
        ACTIVE_CONNECTIONS.dec()


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    if not settings.DEBUG:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging and metrics"""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_response(
            success=False,
            message=exc.detail,
            status_code=exc.status_code
        )
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.warning(
        "Validation error",
        errors=exc.errors(),
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=422,
        content=create_response(
            success=False,
            message="Validation error",
            status_code=422,
            data={"validation_errors": exc.errors()}
        )
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        "Unexpected error occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content=create_response(
            success=False,
            message="Internal server error" if not settings.DEBUG else str(exc),
            status_code=500
        )
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return create_response(
        success=True,
        message="Service is healthy",
        data={
            "version": settings.VERSION,
            "status": "healthy",
            "timestamp": time.time()
        }
    )


# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return create_response(
        success=True,
        message="Water-Borne Disease Monitoring API",
        data={
            "version": settings.VERSION,
            "api_version": settings.API_VERSION,
            "status": "running",
            "features": {
                "authentication": True,
                "rate_limiting": settings.RATE_LIMIT_ENABLED,
                "caching": settings.CACHE_ENABLED,
                "monitoring": settings.ENABLE_METRICS,
                "ml_predictions": settings.ENABLE_ML_PREDICTIONS,
                "background_tasks": settings.ENABLE_BACKGROUND_TASKS
            },
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "geographic_coverage": "Northeast India"
        }
    )


# Cache status endpoint
@app.get("/cache/status")
async def cache_status():
    """Get cache system status"""
    if not settings.CACHE_ENABLED:
        return create_response(
            success=True,
            message="Cache is disabled",
            data={"enabled": False}
        )
    
    try:
        # Test Redis connection
        await redis_client.ping()
        info = await redis_client.info()
        
        return create_response(
            success=True,
            message="Cache is operational",
            data={
                "enabled": True,
                "connected": True,
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed")
            }
        )
    except Exception as e:
        logger.error("Cache health check failed", error=str(e))
        return create_response(
            success=False,
            message="Cache system error",
            data={"enabled": True, "connected": False, "error": str(e)}
        )


# Include routers with rate limiting
if settings.RATE_LIMIT_ENABLED:
    # Apply specific rate limits to different router groups
    routes_auth.router.dependencies = []  # Auth has its own rate limiting
    routes_data.router.dependencies = []
    routes_alerts.router.dependencies = []
    routes_ml.router.dependencies = []

# Include all routers
app.include_router(
    routes_auth.router, 
    prefix=f"{settings.API_V1_STR}/auth", 
    tags=["Authentication"]
)
app.include_router(
    routes_data.router, 
    prefix=f"{settings.API_V1_STR}/data", 
    tags=["Data Management"]
)
app.include_router(
    routes_alerts.router, 
    prefix=f"{settings.API_V1_STR}/alerts", 
    tags=["Alert System"]
)
app.include_router(
    routes_ml.router, 
    prefix=f"{settings.API_V1_STR}/ml", 
    tags=["Machine Learning"]
)

# Include health check router if enabled
if settings.HEALTH_CHECK_ENABLED:
    app.include_router(
        health_check_router, 
        prefix="/health", 
        tags=["Health Checks"]
    )


# Startup event for additional initialization
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("Performing additional startup tasks")
    
    # Schedule background tasks if enabled
    if settings.ENABLE_BACKGROUND_TASKS:
        # You can add periodic tasks here
        pass
    
    # Warm up cache with frequently accessed data
    if settings.CACHE_ENABLED:
        try:
            # Cache frequently accessed reference data
            await warm_up_cache()
        except Exception as e:
            logger.warning("Cache warm-up failed", error=str(e))


async def warm_up_cache():
    """Warm up cache with frequently accessed data"""
    logger.info("Warming up cache")
    
    try:
        # Cache system configuration
        await get_cache().set(
            "system:config",
            {
                "supported_languages": settings.SUPPORTED_LANGUAGES,
                "geo_bounds": settings.GEO_BOUNDS,
                "version": settings.VERSION
            },
            ttl=3600
        )
        
        # Cache alert message templates
        from app.core.utils import ALERT_MESSAGES
        await get_cache().set(
            "alert_messages",
            ALERT_MESSAGES,
            ttl=3600
        )
        
        logger.info("Cache warm-up completed successfully")
        
    except Exception as e:
        logger.error("Cache warm-up failed", error=str(e))



# Add this to your existing main.py in the lifespan function

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager for startup and shutdown events"""
    logger.info("Starting Water-Borne Disease Monitoring API")

    # --- Firebase init ---
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT)
            firebase_admin.initialize_app(cred)
        logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.error("Firebase initialization failed", error=str(e))
        raise

    # --- Redis init ---
    try:
        redis = await init_redis()
        if redis:
            logger.info("Redis connection established successfully")
        else:
            logger.warning("Redis disabled or not available")
    except Exception as e:
        logger.error("Redis initialization failed", error=str(e))
        if settings.CACHE_ENABLED:
            logger.warning("Continuing without cache functionality")

    # --- Firestore init ---
    try:
        await firestore_service.query_collection("users", limit=1)
        logger.info("Firestore connection verified")
    except Exception as e:
        logger.error("Firestore connection failed", error=str(e))
        raise

    # --- ML model init ---
    try:
        from app.ml.model import initialize_models
        success = initialize_models()
        if success:
            logger.info("ML models initialized successfully")
            
            # Log which models are loaded
            from app.ml.model import get_model
            model = get_model()
            loaded_models = []
            if model.water_model.is_loaded:
                loaded_models.append("water_quality_model")
            if model.health_model.is_loaded:
                loaded_models.append("health_risk_model")
            
            if loaded_models:
                logger.info(f"Loaded models: {', '.join(loaded_models)}")
            else:
                logger.warning("No ML models loaded - predictions will use fallback methods")
        else:
            logger.warning("ML model initialization failed - predictions will use fallback methods")
            
    except Exception as e:
        logger.warning("ML model initialization encountered errors", error=str(e))
        logger.info("Application will continue with limited ML functionality")

    logger.info("Application startup completed successfully")

    # --- Yield control to FastAPI ---
    yield

    # --- Cleanup on shutdown ---
    logger.info("Shutting down application")

    try:
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")
    except Exception as e:
        logger.error("Error closing Redis connection", error=str(e))

    logger.info("Application shutdown completed")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn_config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "log_config": None,  # Use our structured logging
        "access_log": False,  # Handled by our middleware
        "workers": settings.MAX_WORKERS if not settings.DEBUG else 1,
        "reload": settings.DEBUG,
    }
    
    uvicorn.run(**uvicorn_config)