"""
Main application entry point for FlashCAMP backend.
Initializes the FastAPI application and routes.
"""
import os
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from .app.routes import router as api_router
from .database import create_db_and_tables
from .app.engines import ml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] [-] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("flashcamp.backend.main")

# Initialize FastAPI application
app = FastAPI(
    title="FlashDNA API",
    description="Backend API for FlashCAMP Startup Analysis Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    # Create database tables if they don't exist
    create_db_and_tables()
    
    # Ensure ML models are loaded
    ml._load_model()
    
    # Log application startup
    mode = "development" if os.environ.get("ENV") == "dev" else "production"
    logger.info(f"FlashDNA API initialized in {mode} mode")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 