"""
Prometheus metrics for FlashCAMP backend
Provides metrics collection for monitoring and alerting
"""
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import time
import os
import platform
import sys
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

# Application Info
APP_INFO = Info(
    "flashcamp_app_info",
    "Application information including version and environment"
)

# Initialize with basic info
APP_INFO.info({
    "version": os.environ.get("APP_VERSION", "1.0.0"),
    "environment": os.environ.get("ENVIRONMENT", "development"),
    "python_version": platform.python_version(),
    "platform": platform.platform()
})

# Request metrics
REQUEST_COUNT = Counter(
    "flashcamp_requests_total",
    "Total count of requests by endpoint and method",
    ["endpoint", "method", "status"]
)

REQUEST_LATENCY = Histogram(
    "flashcamp_request_latency_seconds",
    "Request latency in seconds by endpoint and method",
    ["endpoint", "method"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

# Business metrics
ANALYSES_COUNT = Counter(
    "flashcamp_analyses_total",
    "Total number of startup analyses performed",
    ["startup_stage", "success_prediction"]
)

REPORT_GENERATION_COUNT = Counter(
    "flashcamp_reports_generated_total",
    "Total number of reports generated",
    ["report_type"]
)

REPORT_GENERATION_TIME = Histogram(
    "flashcamp_report_generation_seconds",
    "Time taken to generate reports",
    ["report_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

# Score distribution metrics
PILLAR_SCORE_DISTRIBUTION = Histogram(
    "flashcamp_pillar_score_distribution",
    "Distribution of pillar scores",
    ["pillar"],
    buckets=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
)

SUCCESS_PROBABILITY_DISTRIBUTION = Histogram(
    "flashcamp_success_probability_distribution",
    "Distribution of success probability predictions",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Alerts metrics
ALERTS_GENERATED = Counter(
    "flashcamp_alerts_generated_total",
    "Total number of alerts generated",
    ["alert_type", "severity"]
)

# System metrics
ACTIVE_REQUESTS = Gauge(
    "flashcamp_active_requests",
    "Number of active requests currently being processed"
)

MODEL_PREDICTION_TIME = Histogram(
    "flashcamp_model_prediction_seconds",
    "Time taken for model prediction",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

MODEL_LOADED = Counter(
    "flashcamp_model_loaded",
    "Count of successful model loads",
    ["model_path"]
)

MODEL_LOAD_ERRORS = Counter(
    "flashcamp_model_load_errors_total",
    "Count of model loading errors",
    ["model_path", "error_type"]
)

ERROR_COUNT = Counter(
    "flashcamp_errors_total",
    "Total number of errors by type",
    ["error_type", "endpoint"]
)

VALIDATION_ERRORS = Counter(
    "flashcamp_validation_errors_total",
    "Count of validation errors by field",
    ["field"]
)

# Database metrics
DB_QUERY_TIME = Histogram(
    "flashcamp_db_query_seconds",
    "Time taken for database queries",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

DB_CONNECTION_ERRORS = Counter(
    "flashcamp_db_connection_errors_total",
    "Count of database connection errors"
)

# Memory usage metrics
MEMORY_USAGE = Gauge(
    "flashcamp_memory_usage_bytes",
    "Memory usage of the application in bytes"
)

def update_memory_usage():
    """Update memory usage metric"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        MEMORY_USAGE.set(memory_info.rss)
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to update memory usage metric: {e}")

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware that collects Prometheus metrics for requests
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exclude metrics endpoint from metrics collection to avoid infinite loops
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Track active requests
        ACTIVE_REQUESTS.inc()
        
        # Get the endpoint and method
        endpoint = request.url.path
        method = request.method
        
        # Time the request
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Record request count and latency
            REQUEST_COUNT.labels(
                endpoint=endpoint,
                method=method,
                status=response.status_code
            ).inc()
            
            REQUEST_LATENCY.labels(
                endpoint=endpoint,
                method=method
            ).observe(time.time() - start_time)
            
            # Periodically update memory usage (every ~100 requests)
            if id(request) % 100 == 0:
                update_memory_usage()
            
            return response
        except Exception as e:
            # Record errors
            ERROR_COUNT.labels(
                error_type=type(e).__name__,
                endpoint=endpoint
            ).inc()
            
            REQUEST_COUNT.labels(
                endpoint=endpoint,
                method=method,
                status=500
            ).inc()
            
            raise
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.dec()

# Context manager for timing code blocks
class TimerContextManager:
    """Context manager for timing code blocks and recording metrics"""
    def __init__(self, metric, labels=None):
        self.metric = metric
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:  # Only record successful operations
            duration = time.time() - self.start_time
            if self.labels:
                self.metric.labels(**self.labels).observe(duration)
            else:
                self.metric.observe(duration)

# Helper functions for tracking metrics

def track_analysis(startup_stage: str = "unknown", success_prediction: str = "unknown"):
    """
    Track a startup analysis
    
    Args:
        startup_stage: The funding stage of the startup (e.g., "Seed", "Series A")
        success_prediction: The success prediction range ("high", "medium", "low")
    """
    ANALYSES_COUNT.labels(
        startup_stage=startup_stage,
        success_prediction=success_prediction
    ).inc()

def track_report_generation(report_type: str = "standard"):
    """
    Get context manager for tracking report generation time
    
    Args:
        report_type: The type of report being generated
    """
    REPORT_GENERATION_COUNT.labels(report_type=report_type).inc()
    return TimerContextManager(REPORT_GENERATION_TIME, {"report_type": report_type})

def track_model_prediction(model_name: str):
    """Get context manager for tracking model prediction time"""
    return TimerContextManager(MODEL_PREDICTION_TIME, {"model_name": model_name})

def track_error(error_type: str, endpoint: str = "unknown"):
    """Track an error"""
    ERROR_COUNT.labels(error_type=error_type, endpoint=endpoint).inc()

def track_model_loaded(model_path: str):
    """Track successful model load"""
    MODEL_LOADED.labels(model_path=model_path).inc()

def track_model_load_error(model_path: str, error_type: str):
    """Track model loading error"""
    MODEL_LOAD_ERRORS.labels(model_path=model_path, error_type=error_type).inc()

def track_pillar_score(pillar: str, score: float):
    """Track pillar score distribution"""
    try:
        if 0 <= score <= 10:
            PILLAR_SCORE_DISTRIBUTION.labels(pillar=pillar).observe(score)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to track pillar score: {e}")

def track_success_probability(probability: float):
    """Track success probability distribution"""
    try:
        if 0 <= probability <= 1:
            SUCCESS_PROBABILITY_DISTRIBUTION.observe(probability)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to track success probability: {e}")

def track_alert(alert_type: str, severity: str = "warning"):
    """Track an alert generation"""
    ALERTS_GENERATED.labels(alert_type=alert_type, severity=severity).inc()

def track_validation_error(field: str):
    """Track a validation error"""
    VALIDATION_ERRORS.labels(field=field).inc()

def track_db_query(operation: str):
    """Get context manager for tracking database query time"""
    return TimerContextManager(DB_QUERY_TIME, {"operation": operation})

def track_db_error():
    """Track a database error"""
    DB_CONNECTION_ERRORS.inc() 