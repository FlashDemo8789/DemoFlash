"""
CRUD operations for database models.
This module provides database operations (Create, Read, Update, Delete) for the application models.
"""
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any, Union
import datetime
import json

from .models import Startup, Analysis, Report, User


# Startup CRUD operations
def get_startup(db: Session, startup_id: int) -> Optional[Startup]:
    """Get a startup by ID"""
    return db.query(Startup).filter(Startup.id == startup_id).first()


def get_startup_by_name(db: Session, name: str) -> Optional[Startup]:
    """Get a startup by name"""
    return db.query(Startup).filter(Startup.name == name).first()


def get_startup_by_startup_id(db: Session, startup_id: str) -> Optional[Startup]:
    """Get a startup by its external ID"""
    return db.query(Startup).filter(Startup.startup_id == startup_id).first()


def get_startups(
    db: Session, 
    skip: int = 0, 
    limit: int = 100, 
    sector: Optional[str] = None
) -> List[Startup]:
    """Get all startups with optional filtering"""
    query = db.query(Startup)
    
    if sector:
        query = query.filter(Startup.sector == sector)
        
    return query.offset(skip).limit(limit).all()


def create_startup(db: Session, name: str, data: Dict[str, Any]) -> Startup:
    """Create a new startup"""
    db_startup = Startup(
        name=name,
        startup_id=data.get("startup_id", ""),
        sector=data.get("sector", ""),
        subsector=data.get("subsector", ""),
        country=data.get("country", ""),
        website=data.get("website", ""),
        founding_year=data.get("founding_year"),
        funding_stage=data.get("funding_stage", ""),
        industry=data.get("industry", ""),
        metrics_data=data
    )
    db.add(db_startup)
    db.commit()
    db.refresh(db_startup)
    return db_startup


def update_startup(db: Session, startup_id: int, data: Dict[str, Any]) -> Optional[Startup]:
    """Update an existing startup"""
    db_startup = get_startup(db, startup_id)
    if not db_startup:
        return None
        
    for key, value in data.items():
        if hasattr(db_startup, key):
            setattr(db_startup, key, value)
    
    # Update the metrics data JSON with all received data
    if db_startup.metrics_data:
        metrics_data = db_startup.metrics_data
        metrics_data.update(data)
        db_startup.metrics_data = metrics_data
    else:
        db_startup.metrics_data = data
        
    db_startup.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(db_startup)
    return db_startup


def delete_startup(db: Session, startup_id: int) -> bool:
    """Delete a startup"""
    db_startup = get_startup(db, startup_id)
    if not db_startup:
        return False
        
    db.delete(db_startup)
    db.commit()
    return True


# Analysis CRUD operations
def get_analysis(db: Session, analysis_id: int) -> Optional[Analysis]:
    """Get an analysis by ID"""
    return db.query(Analysis).filter(Analysis.id == analysis_id).first()


def get_analyses_for_startup(
    db: Session, 
    startup_id: int, 
    skip: int = 0, 
    limit: int = 100
) -> List[Analysis]:
    """Get all analyses for a specific startup"""
    return db.query(Analysis).filter(
        Analysis.startup_id == startup_id
    ).offset(skip).limit(limit).all()


def create_analysis(
    db: Session, 
    startup_id: int, 
    results: Dict[str, Any]
) -> Analysis:
    """Create a new analysis record"""
    db_analysis = Analysis(
        startup_id=startup_id,
        capital_score=results.get("capital_score", 0.5),
        advantage_score=results.get("advantage_score", 0.5),
        market_score=results.get("market_score", 0.5),
        people_score=results.get("people_score", 0.5),
        overall_score=results.get("overall_score", 0.5),
        success_probability=results.get("success_probability", 0.5),
        alerts=results.get("alerts", []),
        analysis_version="1.0"  # Set current version
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis


# Report CRUD operations
def get_report(db: Session, report_id: int) -> Optional[Report]:
    """Get a report by ID"""
    return db.query(Report).filter(Report.id == report_id).first()


def get_reports_for_startup(
    db: Session, 
    startup_id: int, 
    skip: int = 0, 
    limit: int = 100
) -> List[Report]:
    """Get all reports for a specific startup"""
    return db.query(Report).filter(
        Report.startup_id == startup_id
    ).offset(skip).limit(limit).all()


def create_report(
    db: Session, 
    startup_id: int, 
    report_name: str,
    report_type: str,
    file_path: str,
    created_by: Optional[int] = None
) -> Report:
    """Create a new report record"""
    db_report = Report(
        startup_id=startup_id,
        report_name=report_name,
        report_type=report_type,
        file_path=file_path,
        created_by=created_by
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report


# User CRUD operations
def get_user(db: Session, user_id: int) -> Optional[User]:
    """Get a user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get a user by username"""
    return db.query(User).filter(User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Get all users"""
    return db.query(User).offset(skip).limit(limit).all()


def create_user(
    db: Session, 
    email: str, 
    username: str, 
    hashed_password: str,
    full_name: Optional[str] = None,
    is_admin: bool = False
) -> User:
    """Create a new user"""
    db_user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        full_name=full_name,
        is_admin=is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user 