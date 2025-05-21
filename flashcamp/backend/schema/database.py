"""
Database connection module.
This module provides SQLAlchemy session and engine setup for the application.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import os

from ..config import get_settings, settings

# Create SQLAlchemy engine and session factory
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# Create engine with appropriate arguments based on URL
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import models to ensure they're registered with the metadata
from .models import Base, Startup, Analysis, Report, User


def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize the database with tables and default data if needed"""
    create_tables()
    
    # Add any initial data seeding here if needed
    # For example, creating an admin user on first setup
    # This function can be called from the app's startup event 