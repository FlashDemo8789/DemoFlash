"""
Configuration settings for the FlashDNA application.
This module loads settings from environment variables and provides 
them to the application through Pydantic BaseSettings.
"""
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from functools import lru_cache

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Try to load the .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    # Try loading from a different location as fallback
    load_dotenv()  # Try default locations

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    APP_NAME: str = "FlashDNA"
    VERSION: str = "1.0.0"
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS settings
    CORS_ORIGINS_STR: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", "20"))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "temporary_dev_key_replace_in_production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/")
    DATA_PATH: str = os.getenv("DATA_PATH", "data/gold/")
    FLASHDNA_MODEL: str = os.getenv("FLASHDNA_MODEL", "models/success_xgb.joblib")
    
    # Report settings
    REPORT_TEMPLATE_PATH: str = os.getenv("REPORT_TEMPLATE_PATH", "reports/templates/")
    REPORT_OUTPUT_PATH: str = os.getenv("REPORT_OUTPUT_PATH", "reports/pdf/")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    
    # Type hints for methods
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "allow",  # Allow extra fields like CORS_ORIGINS_STR
    }
    
    def get_cors_origins(self) -> List[str]:
        """Parse the CORS_ORIGINS string into a list."""
        if self.ENVIRONMENT == "development":
            # In development, we can be more permissive for convenience
            return ["*"]
        elif self.CORS_ORIGINS_STR:
            # In production, use only the specified origins
            return [origin.strip() for origin in self.CORS_ORIGINS_STR.split(",") if origin.strip()]
        else:
            # Fallback to empty list (no CORS allowed)
            return []
    
    def get_absolute_model_path(self) -> str:
        """Get the absolute path to the model file."""
        # If FLASHDNA_MODEL is already absolute, use it
        if os.path.isabs(self.FLASHDNA_MODEL):
            return self.FLASHDNA_MODEL
        # Otherwise, resolve relative to the project root
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), self.FLASHDNA_MODEL)
    
    def get_database_args(self) -> Dict[str, Any]:
        """Get database connection arguments based on the URL scheme."""
        if self.DATABASE_URL.startswith("sqlite"):
            # SQLite doesn't use pool settings
            return {"connect_args": {"check_same_thread": False}}
        else:
            # PostgreSQL, MySQL, etc.
            return {
                "pool_size": self.DATABASE_POOL_SIZE,
                "max_overflow": self.DATABASE_MAX_OVERFLOW
            }

@lru_cache()
def get_settings() -> Settings:
    """Create and cache a Settings instance."""
    return Settings()

# Export a singleton instance for easier imports
settings = get_settings() 