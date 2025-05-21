"""
Database module for FlashCAMP backend.
Contains SQLAlchemy setup, session management, and database models.
"""
import os
import logging
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# Configure logging
logger = logging.getLogger(__name__)

# Database URL from environment or default to SQLite
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./flashcamp.db")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()

# Database Models
class Startup(Base):
    """Startup model for database storage"""
    __tablename__ = "startups"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    startup_id = Column(String(50), unique=True, index=True)
    sector = Column(String(100))
    subsector = Column(String(100))
    country = Column(String(100))
    website = Column(String(255))
    founding_year = Column(Integer)
    funding_stage = Column(String(50))
    industry = Column(String(100))
    metrics_data = Column(JSON)  # Store all metrics as JSON
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Relationships
    analyses = relationship("Analysis", back_populates="startup")
    reports = relationship("Report", back_populates="startup")

class Analysis(Base):
    """Analysis results model"""
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    startup_id = Column(Integer, ForeignKey("startups.id"), nullable=False)
    capital_score = Column(Float)
    advantage_score = Column(Float)
    market_score = Column(Float)
    people_score = Column(Float)
    overall_score = Column(Float)
    success_probability = Column(Float)
    alerts = Column(JSON)  # Store alerts as JSON
    analysis_version = Column(String(50))  # Version of analysis/model used
    created_at = Column(DateTime)
    
    # Relationships
    startup = relationship("Startup", back_populates="analyses")

class User(Base):
    """User model for authentication and report generation"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean)
    is_admin = Column(Boolean)
    created_at = Column(DateTime)
    last_login = Column(DateTime)
    
    # Relationships
    reports = relationship("Report", back_populates="created_by_user")

class Report(Base):
    """Report model for generated reports"""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    startup_id = Column(Integer, ForeignKey("startups.id"), nullable=False)
    report_name = Column(String(255))
    report_type = Column(String(50))  # PDF, Excel, etc.
    file_path = Column(String(512))  # Path to stored report
    created_at = Column(DateTime)
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    startup = relationship("Startup", back_populates="reports")
    created_by_user = relationship("User", back_populates="reports")

# Database dependency for route functions
def get_db():
    """
    Get database session dependency for FastAPI routes.
    Used with Depends in routes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_db_and_tables():
    """Create database tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise 