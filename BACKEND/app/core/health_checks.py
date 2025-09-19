from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import asyncio
import time
from datetime import datetime, timezone
from redis import asyncio as aioredis  # if you actually need redis here

import firebase_admin
from firebase_admin import auth

from app.config import settings
from app.core.utils import create_response
from app.core.cache import redis_client, get_cache
from app.db.database import firestore_service
from app.core.monitoring import structured_logger, alerting_system

router = APIRouter()
logger = structured_logger


@router.get("/")
async def basic_health_check():
    """Basic health check endpoint"""
    return create_response(
        success=True,
        message="Health check passed",
        data={
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": settings.VERSION
        }
    )


@router.get("/detailed")
async def detailed_health_check():
    """Comprehensive health check for all system components"""
    try:
        health_status = await alerting_system.check_system_health()
        
        # Determine HTTP status code based on health
        status_code = 200
        if health_status['status'] == 'degraded':
            status_code = 206  # Partial Content
        elif health_status['status'] == 'unhealthy':
            status_code = 503  # Service Unavailable
        
        return JSONResponse(
            status_code=status_code,
            content=create_response(
                success=health_status['status'] != 'unhealthy',
                message=f"System status: {health_status['status']}",
                data=health_status
            )
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content=create_response(
                success=False,
                message="Health check failed",
                data={"error": str(e)}
            )
        )


@router.get("/database")
async def database_health_check():
    """Check database connectivity and performance"""
    start_time = time.time()
    
    try:
        # Test Firestore connection with a simple query
        await firestore_service.query_collection("users", limit=1)
        
        duration = time.time() - start_time
        
        status = "healthy"
        if duration > settings.HEALTH_CHECK_DB_TIMEOUT:
            status = "slow"
        
        return create_response(
            success=True,
            message="Database health check passed",
            data={
                "status": status,
                "response_time_seconds": duration,
                "database_type": "firestore",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("Database health check failed", error=str(e), duration=duration)
        
        return JSONResponse(
            status_code=503,
            content=create_response(
                success=False,
                message="Database health check failed",
                data={
                    "status": "unhealthy",
                    "error": str(e),
                    "response_time_seconds": duration,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        )


@router.get("/cache")
async def cache_health_check():
    """Check Redis cache connectivity and performance"""
    if not settings.CACHE_ENABLED:
        return create_response(
            success=True,
            message="Cache is disabled",
            data={"status": "disabled"}
        )
    
    start_time = time.time()
    
    try:
        cache = get_cache()
        
        # Test basic operations
        test_key = "health_check_test"
        test_value = {"timestamp": time.time()}
        
        # Test set operation
        await cache.set(test_key, test_value, ttl=10)
        
        # Test get operation
        retrieved_value = await cache.get(test_key)
        
        # Test delete operation
        await cache.delete(test_key)
        
        duration = time.time() - start_time
        
        # Get cache info
        cache_info = await cache.get_info()
        
        status = "healthy"
        if duration > settings.HEALTH_CHECK_REDIS_TIMEOUT:
            status = "slow"
        
        return create_response(
            success=True,
            message="Cache health check passed",
            data={
                "status": status,
                "response_time_seconds": duration,
                "operations_tested": ["set", "get", "delete"],
                "cache_info": cache_info,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("Cache health check failed", error=str(e), duration=duration)
        
        return JSONResponse(
            status_code=503,
            content=create_response(
                success=False,
                message="Cache health check failed",
                data={
                    "status": "unhealthy",
                    "error": str(e),
                    "response_time_seconds": duration,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        )


@router.get("/firebase")
async def firebase_health_check():
    """Check Firebase Authentication service connectivity"""
    start_time = time.time()
    
    try:
        # Test Firebase connection by listing users (with minimal results)
        users_page = auth.list_users(max_results=1)
        
        duration = time.time() - start_time
        
        status = "healthy"
        if duration > settings.HEALTH_CHECK_FIREBASE_TIMEOUT:
            status = "slow"
        
        return create_response(
            success=True,
            message="Firebase health check passed",
            data={
                "status": status,
                "response_time_seconds": duration,
                "service": "firebase_auth",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("Firebase health check failed", error=str(e), duration=duration)
        
        return JSONResponse(
            status_code=503,
            content=create_response(
                success=False,
                message="Firebase health check failed",
                data={
                    "status": "unhealthy",
                    "error": str(e),
                    "response_time_seconds": duration,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        )


@router.get("/resources")
async def resource_health_check():
    """Check system resource usage"""
    try:
        import psutil
        
        # Get system resource information
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # Get process-specific information
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()
        
        # Determine status based on thresholds
        status = "healthy"
        warnings = []
        
        if memory.percent > 80:
            status = "warning"
            warnings.append(f"High memory usage: {memory.percent:.1f}%")
        
        if cpu_percent > 80:
            status = "warning" 
            warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if disk.percent > 90:
            status = "warning"
            warnings.append(f"High disk usage: {disk.percent:.1f}%")
        
        resource_data = {
            "status": status,
            "warnings": warnings,
            "system": {
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / 1024 / 1024,
                "cpu_percent": cpu_percent,
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024
            },
            "process": {
                "memory_mb": process_memory.rss / 1024 / 1024,
                "cpu_percent": process_cpu,
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return create_response(
            success=True,
            message="Resource health check completed",
            data=resource_data
        )
        
    except ImportError:
        return create_response(
            success=False,
            message="Resource monitoring not available - psutil not installed",
            data={"status": "unavailable"}
        )
    except Exception as e:
        logger.error("Resource health check failed", error=str(e))
        return create_response(
            success=False,
            message="Resource health check failed",
            data={"status": "error", "error": str(e)}
        )


@router.get("/dependencies")
async def dependencies_health_check():
    """Check all external dependencies"""
    dependencies = {
        "database": {"status": "unknown", "details": {}},
        "cache": {"status": "unknown", "details": {}}, 
        "firebase": {"status": "unknown", "details": {}},
        "resources": {"status": "unknown", "details": {}}
    }
    
    overall_status = "healthy"
    
    # Check database
    try:
        start_time = time.time()
        await firestore_service.query_collection("users", limit=1)
        duration = time.time() - start_time
        
        dependencies["database"] = {
            "status": "healthy" if duration < 5.0 else "slow",
            "details": {"response_time": duration}
        }
    except Exception as e:
        dependencies["database"] = {
            "status": "unhealthy",
            "details": {"error": str(e)}
        }
        overall_status = "degraded"
    
    # Check cache
    if settings.CACHE_ENABLED:
        try:
            cache = get_cache()
            await cache.get("test_key")
            cache_info = await cache.get_info()
            
            dependencies["cache"] = {
                "status": "healthy" if cache_info.get("connected") else "unhealthy",
                "details": cache_info
            }
        except Exception as e:
            dependencies["cache"] = {
                "status": "unhealthy", 
                "details": {"error": str(e)}
            }
            overall_status = "degraded"
    else:
        dependencies["cache"] = {
            "status": "disabled",
            "details": {"message": "Cache is disabled"}
        }
    
    # Check Firebase
    try:
        auth.list_users(max_results=1)
        dependencies["firebase"] = {
            "status": "healthy",
            "details": {"service": "firebase_auth"}
        }
    except Exception as e:
        dependencies["firebase"] = {
            "status": "unhealthy",
            "details": {"error": str(e)}
        }
        overall_status = "degraded"
    
    # Check resources
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        resource_status = "healthy"
        if memory.percent > 80 or cpu > 80:
            resource_status = "warning"
            if overall_status == "healthy":
                overall_status = "degraded"
        
        dependencies["resources"] = {
            "status": resource_status,
            "details": {
                "memory_percent": memory.percent,
                "cpu_percent": cpu
            }
        }
    except ImportError:
        dependencies["resources"] = {
            "status": "unavailable",
            "details": {"message": "psutil not available"}
        }
    
    status_code = 200
    if overall_status == "degraded":
        status_code = 206
    elif overall_status == "unhealthy":
        status_code = 503
    
    return JSONResponse(
        status_code=status_code,
        content=create_response(
            success=overall_status != "unhealthy",
            message=f"Dependencies check completed - Status: {overall_status}",
            data={
                "overall_status": overall_status,
                "dependencies": dependencies,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    )


@router.get("/readiness")
async def readiness_check():
    """Kubernetes-style readiness probe"""
    try:
        # Check critical dependencies for readiness
        checks = []
        ready = True
        
        # Database check
        try:
            await asyncio.wait_for(
                firestore_service.query_collection("users", limit=1),
                timeout=5.0
            )
            checks.append({"name": "database", "status": "ready"})
        except asyncio.TimeoutError:
            checks.append({"name": "database", "status": "timeout"})
            ready = False
        except Exception as e:
            checks.append({"name": "database", "status": "error", "error": str(e)})
            ready = False
        
        # Cache check (if enabled)
        if settings.CACHE_ENABLED:
            try:
                cache = get_cache()
                await asyncio.wait_for(cache.get("readiness_test"), timeout=2.0)
                checks.append({"name": "cache", "status": "ready"})
            except asyncio.TimeoutError:
                checks.append({"name": "cache", "status": "timeout"}) 
                ready = False
            except Exception as e:
                checks.append({"name": "cache", "status": "error", "error": str(e)})
                ready = False
        
        # Firebase check
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: auth.list_users(max_results=1)
                ),
                timeout=10.0
            )
            checks.append({"name": "firebase", "status": "ready"})
        except asyncio.TimeoutError:
            checks.append({"name": "firebase", "status": "timeout"})
            ready = False
        except Exception as e:
            checks.append({"name": "firebase", "status": "error", "error": str(e)})
            ready = False
        
        status_code = 200 if ready else 503
        
        return JSONResponse(
            status_code=status_code,
            content=create_response(
                success=ready,
                message="Readiness check completed",
                data={
                    "ready": ready,
                    "checks": checks,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        )
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content=create_response(
                success=False,
                message="Readiness check failed",
                data={"error": str(e)}
            )
        )


@router.get("/liveness")
async def liveness_check():
    """Kubernetes-style liveness probe"""
    try:
        # Basic application liveness - just check if we can respond
        return create_response(
            success=True,
            message="Application is alive",
            data={
                "alive": True,
                "version": settings.VERSION,
                "uptime_seconds": time.time() - getattr(liveness_check, 'start_time', time.time()),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    except Exception as e:
        logger.error("Liveness check failed", error=str(e))
        return JSONResponse(
            status_code=500,
            content=create_response(
                success=False,
                message="Liveness check failed",
                data={"error": str(e)}
            )
        )


# Set start time for uptime calculation
liveness_check.start_time = time.time()


health_check_router = router