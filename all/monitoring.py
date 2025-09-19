import structlog
import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import json
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
import asyncio
import time
from functools import wraps

from app.config import settings

# Prometheus metrics registry
REGISTRY = CollectorRegistry()

# Application metrics
APP_INFO = Info(
    'app_info',
    'Application information',
    registry=REGISTRY
)

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code', 'user_role'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

ACTIVE_CONNECTIONS = Gauge(
    'http_active_connections',
    'Number of active HTTP connections',
    registry=REGISTRY
)

DATABASE_OPERATIONS = Counter(
    'database_operations_total',
    'Total database operations',
    ['operation', 'collection', 'status'],
    registry=REGISTRY
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'status'],
    registry=REGISTRY
)

ML_PREDICTIONS = Counter(
    'ml_predictions_total',
    'Total ML predictions made',
    ['prediction_type', 'risk_level', 'model_version'],
    registry=REGISTRY
)

ALERTS_SENT = Counter(
    'alerts_sent_total',
    'Total alerts sent',
    ['alert_type', 'severity', 'channel'],
    registry=REGISTRY
)

USER_REGISTRATIONS = Counter(
    'user_registrations_total',
    'Total user registrations',
    ['role', 'state'],
    registry=REGISTRY
)

DATA_UPLOADS = Counter(
    'data_uploads_total',
    'Total data uploads',
    ['data_type', 'source', 'status'],
    registry=REGISTRY
)

BACKGROUND_TASKS = Counter(
    'background_tasks_total',
    'Total background tasks executed',
    ['task_type', 'status'],
    registry=REGISTRY
)

RATE_LIMIT_HITS = Counter(
    'rate_limit_hits_total',
    'Total rate limit hits',
    ['endpoint', 'user_type'],
    registry=REGISTRY
)

# System metrics
MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    registry=REGISTRY
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    registry=REGISTRY
)

REDIS_CONNECTIONS = Gauge(
    'redis_connections_active',
    'Active Redis connections',
    registry=REGISTRY
)

FIREBASE_OPERATIONS = Counter(
    'firebase_operations_total',
    'Total Firebase operations',
    ['operation', 'status'],
    registry=REGISTRY
)


class StructuredLogger:
    """Custom structured logger with contextual information"""

    def __init__(self):
        self.logger = structlog.get_logger()

    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self._log("error", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self._log("debug", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self._log("critical", message, **kwargs)

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method"""
        # Add standard context
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "health_monitor",
            "version": settings.VERSION,
            **kwargs
        }

        log_func = getattr(self.logger, level)
        log_func(message, **context)


def setup_logging():
    """Setup structured logging configuration"""
    if settings.STRUCTURED_LOGGING:
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if not settings.DEBUG
                else structlog.dev.ConsoleRenderer(colors=True)
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if settings.DEBUG
        else '%(message)s',
        stream=sys.stdout
    )

    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


def setup_metrics():
    """Initialize Prometheus metrics with application info"""
    APP_INFO.info({
        'version': settings.VERSION,
        'environment': 'development' if settings.DEBUG else 'production',
        'features': json.dumps({
            'cache_enabled': settings.CACHE_ENABLED,
            'rate_limiting': settings.RATE_LIMIT_ENABLED,
            'ml_predictions': settings.ENABLE_ML_PREDICTIONS,
            'background_tasks': settings.ENABLE_BACKGROUND_TASKS
        })
    })


class MetricsCollector:
    """Collect and expose application metrics"""

    @staticmethod
    def record_request(method: str, endpoint: str, status_code: int, duration: float, user_role: str = "anonymous"):
        """Record HTTP request metrics"""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            user_role=user_role
        ).inc()

        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    @staticmethod
    def record_database_operation(operation: str, collection: str, status: str):
        """Record database operation metrics"""
        DATABASE_OPERATIONS.labels(
            operation=operation,
            collection=collection,
            status=status
        ).inc()

    @staticmethod
    def record_cache_operation(operation: str, status: str):
        """Record cache operation metrics"""
        CACHE_OPERATIONS.labels(
            operation=operation,
            status=status
        ).inc()

    @staticmethod
    def record_ml_prediction(prediction_type: str, risk_level: str, model_version: str):
        """Record ML prediction metrics"""
        ML_PREDICTIONS.labels(
            prediction_type=prediction_type,
            risk_level=risk_level,
            model_version=model_version
        ).inc()

    @staticmethod
    def record_alert_sent(alert_type: str, severity: str, channel: str):
        """Record alert sent metrics"""
        ALERTS_SENT.labels(
            alert_type=alert_type,
            severity=severity,
            channel=channel
        ).inc()

    @staticmethod
    def record_user_registration(role: str, state: str):
        """Record user registration metrics"""
        USER_REGISTRATIONS.labels(
            role=role,
            state=state
        ).inc()

    @staticmethod
    def record_data_upload(data_type: str, source: str, status: str):
        """Record data upload metrics"""
        DATA_UPLOADS.labels(
            data_type=data_type,
            source=source,
            status=status
        ).inc()

    @staticmethod
    def record_background_task(task_type: str, status: str):
        """Record background task execution"""
        BACKGROUND_TASKS.labels(
            task_type=task_type,
            status=status
        ).inc()

    @staticmethod
    def record_rate_limit_hit(endpoint: str, user_type: str):
        """Record rate limit hit"""
        RATE_LIMIT_HITS.labels(
            endpoint=endpoint,
            user_type=user_type
        ).inc()

    @staticmethod
    def record_firebase_operation(operation: str, status: str):
        """Record Firebase operation"""
        FIREBASE_OPERATIONS.labels(
            operation=operation,
            status=status
        ).inc()

    @staticmethod
    def update_connection_count(count: int):
        """Update active connection count"""
        ACTIVE_CONNECTIONS.set(count)

    @staticmethod
    def update_memory_usage(bytes_used: int):
        """Update memory usage metric"""
        MEMORY_USAGE.set(bytes_used)

    @staticmethod
    def update_cpu_usage(percentage: float):
        """Update CPU usage percentage"""
        CPU_USAGE.set(percentage)

    @staticmethod
    def update_redis_connections(count: int):
        """Update Redis connection count"""
        REDIS_CONNECTIONS.set(count)


class PerformanceMonitor:
    """Monitor and log performance metrics"""

    def __init__(self):
        self.logger = StructuredLogger()
    
    # FIX 1: Renamed 'time_function' to 'track_task' and 'func_name' to 'task_name'
    def track_task(self, task_name: str):
        """Decorator to time and track Celery task execution"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    # FIX 2: Record success metric
                    metrics_collector.record_background_task(task_type=task_name, status="success")
                    duration = time.time() - start_time
                    
                    if duration > 1.0:  # Log slow operations
                        self.logger.warning(
                            "Slow task detected",
                            task=task_name,
                            duration=duration,
                        )
                    
                    return result
                
                except Exception as e:
                    duration = time.time() - start_time
                    # FIX 3: Record failure metric
                    metrics_collector.record_background_task(task_type=task_name, status="failure")
                    self.logger.error(
                        "Task execution failed",
                        task=task_name,
                        duration=duration,
                        error=str(e)
                    )
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    # FIX 2: Record success metric
                    metrics_collector.record_background_task(task_type=task_name, status="success")
                    duration = time.time() - start_time
                    
                    if duration > 1.0:  # Log slow operations
                        self.logger.warning(
                            "Slow task detected",
                            task=task_name,
                            duration=duration,
                        )
                    
                    return result
                
                except Exception as e:
                    duration = time.time() - start_time
                    # FIX 3: Record failure metric
                    metrics_collector.record_background_task(task_type=task_name, status="failure")
                    self.logger.error(
                        "Task execution failed",
                        task=task_name,
                        duration=duration,
                        error=str(e)
                    )
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator

    def log_resource_usage(self):
        """Log current resource usage"""
        try:
            import psutil
            process = psutil.Process()
            
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            self.logger.info(
                "Resource usage",
                memory_mb=memory_info.rss / 1024 / 1024,
                cpu_percent=cpu_percent,
                threads=process.num_threads(),
                open_files=len(process.open_files())
            )
            
            # Update metrics
            MetricsCollector.update_memory_usage(memory_info.rss)
            MetricsCollector.update_cpu_usage(cpu_percent)
            
        except ImportError:
            self.logger.warning("psutil not available for resource monitoring")
        except Exception as e:
            self.logger.error("Failed to collect resource usage", error=str(e))


class AlertingSystem:
    """System for generating operational alerts"""

    def __init__(self):
        self.logger = StructuredLogger()
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'response_time': 5.0,  # 5 seconds
            'memory_usage': 0.8,  # 80% memory usage
            'cpu_usage': 0.8,  # 80% CPU usage
        }

    async def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': {}
        }

        # Check cache connection
        try:
            from app.core.cache import get_cache
            cache = get_cache()
            cache_info = await cache.get_info()
            health_status['checks']['cache'] = {
                'status': 'healthy' if cache_info.get('connected', False) else 'unhealthy',
                'details': cache_info
            }
        except Exception as e:
            health_status['checks']['cache'] = {
                'status': 'unhealthy',
                'error': str(e)
            }

        # Check database connection
        try:
            from app.db.database import firestore_service
            # Simple query to test connection
            await firestore_service.query_collection("users", limit=1)
            health_status['checks']['database'] = {'status': 'healthy'}
        except Exception as e:
            health_status['checks']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'

        # Check Firebase connection
        try:
            import firebase_admin
            from firebase_admin import auth
            # Test Firebase connection
            auth.list_users(max_results=1)
            health_status['checks']['firebase'] = {'status': 'healthy'}
        except Exception as e:
            health_status['checks']['firebase'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'

        # Check resource usage
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100
            cpu_percent = psutil.cpu_percent() / 100
            
            health_status['checks']['resources'] = {
                'status': 'healthy',
                'memory_usage': memory_percent,
                'cpu_usage': cpu_percent
            }
            
            if memory_percent > self.alert_thresholds['memory_usage']:
                health_status['checks']['resources']['status'] = 'warning'
                health_status['status'] = 'degraded'
                self.logger.warning("High memory usage detected", usage=memory_percent)
            
            if cpu_percent > self.alert_thresholds['cpu_usage']:
                health_status['checks']['resources']['status'] = 'warning'
                health_status['status'] = 'degraded'
                self.logger.warning("High CPU usage detected", usage=cpu_percent)
                
        except ImportError:
            health_status['checks']['resources'] = {
                'status': 'unknown',
                'message': 'psutil not available'
            }
        
        return health_status

    def send_alert(self, level: str, message: str, details: Dict[str, Any] = None):
        """Send operational alert"""
        alert_data = {
            'level': level,
            'message': message,
            'service': 'health_monitor',
            'version': settings.VERSION,
            'details': details or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        if level == 'critical':
            self.logger.critical("System alert", **alert_data)
        elif level == 'warning':
            self.logger.warning("System alert", **alert_data)
        else:
            self.logger.info("System alert", **alert_data)


# Initialize monitoring components
structured_logger = StructuredLogger()
metrics_collector = MetricsCollector()
# FIX 4: Renamed instance to 'task_monitor'
task_monitor = PerformanceMonitor()
alerting_system = AlertingSystem()

# Convenience functions
def log_info(message: str, **kwargs):
    """Log info message"""
    structured_logger.info(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Log warning message"""
    structured_logger.warning(message, **kwargs)

def log_error(message: str, **kwargs):
    """Log error message"""
    structured_logger.error(message, **kwargs)

def record_metric(metric_type: str, **kwargs):
    """Record metric based on type"""
    method = getattr(metrics_collector, f'record_{metric_type}', None)
    if method:
        method(**kwargs)
    else:
        log_warning("Unknown metric type", metric_type=metric_type)


# Background task to periodically log system metrics
async def system_metrics_task():
    """Background task to collect and log system metrics"""
    while True:
        try:
            # FIX 5: Use 'task_monitor' to call the method, though the original 'performance_monitor' was also an instance of the same class
            task_monitor.log_resource_usage() 
            health_status = await alerting_system.check_system_health()
            
            if health_status['status'] != 'healthy':
                log_warning("System health check", **health_status)
            
            # Check for any critical issues
            unhealthy_checks = [
                name for name, check in health_status['checks'].items()
                if check.get('status') == 'unhealthy'
            ]
            
            if unhealthy_checks:
                alerting_system.send_alert(
                    'warning',
                    f"Unhealthy system components: {', '.join(unhealthy_checks)}",
                    {'failed_checks': unhealthy_checks}
                )
            
            await asyncio.sleep(60)  # Run every minute
            
        except Exception as e:
            log_error("System metrics task failed", error=str(e))
            await asyncio.sleep(60)