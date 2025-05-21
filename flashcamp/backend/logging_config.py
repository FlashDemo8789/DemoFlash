"""
Logging configuration for FlashCAMP
Sets up structured logging with customizable handlers
"""
import logging
import logging.config
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Import settings
from .config import settings

# Define log directory
LOG_DIR = os.environ.get("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class StructuredJsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record
    """
    def __init__(self, fmt_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize formatter with optional format dictionary
        """
        self.fmt_dict = fmt_dict if fmt_dict else {
            "timestamp": "%(asctime)s",
            "level": "%(levelname)s",
            "message": "%(message)s",
            "module": "%(module)s",
            "function": "%(funcName)s",
            "line": "%(lineno)d",
            "process": "%(process)d",
            "thread": "%(thread)d",
        }
        super(StructuredJsonFormatter, self).__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string
        """
        log_record = {}
        
        # Add format dictionary items
        for key, value in self.fmt_dict.items():
            log_record[key] = self._format_value(value, record)
        
        # Add extra attributes from record
        if hasattr(record, "extra"):
            for key, value in record.extra.items():
                log_record[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_record)
    
    def _format_value(self, fmt: str, record: logging.LogRecord) -> Any:
        """
        Format a value from the record using the log record attributes
        """
        if not isinstance(fmt, str):
            return fmt
        
        if fmt.startswith("%(") and fmt.endswith(")s"):
            attr = fmt[2:-2]
            if hasattr(record, attr):
                return getattr(record, attr)
        
        # Process as regular format string
        try:
            return fmt % record.__dict__
        except:
            return fmt

class RequestIdFilter(logging.Filter):
    """
    Filter that adds request_id to log records if available
    """
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True

# Base logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] [%(name)s] [%(request_id)s] %(message)s"
        },
        "json": {
            "()": "flashcamp.backend.logging_config.StructuredJsonFormatter"
        },
    },
    "filters": {
        "request_id": {
            "()": "flashcamp.backend.logging_config.RequestIdFilter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "filters": ["request_id"],
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filters": ["request_id"],
            "filename": f"{LOG_DIR}/flashcamp.log",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "json",
            "filters": ["request_id"],
            "filename": f"{LOG_DIR}/error.log",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file", "error_file"],
            "level": settings.LOG_LEVEL.upper(),
            "propagate": True
        },
        "backend": {
            "handlers": ["console", "file", "error_file"],
            "level": settings.LOG_LEVEL.upper(),
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        },
        "alembic": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

def setup_logging():
    """
    Configure logging based on the settings
    """
    # Create log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)
    
    logger = logging.getLogger("backend")
    logger.info(f"Logging initialized at level {settings.LOG_LEVEL.upper()}")
    
    return logger

# Function to get a logger
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger configured with the application settings
    """
    return logging.getLogger(name) 