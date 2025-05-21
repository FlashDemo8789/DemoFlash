"""
SQLAlchemy models for the database schema.
These models represent the core data structures for the application.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Startup(Base):
    """Startup entity model"""
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
    
    # Raw metric data as JSON
    metrics_data = Column(JSON)
    
    # Creation and modification timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    analyses = relationship("Analysis", back_populates="startup", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="startup", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Startup(name='{self.name}', id='{self.id}')>"


class Analysis(Base):
    """Analysis entity model representing a scoring/analysis session"""
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    startup_id = Column(Integer, ForeignKey("startups.id"), nullable=False)
    
    # Scores for each pillar
    capital_score = Column(Float)
    advantage_score = Column(Float)
    market_score = Column(Float)
    people_score = Column(Float)
    
    # Overall analysis results
    overall_score = Column(Float)
    success_probability = Column(Float)
    
    # Alerts and insights as JSON
    alerts = Column(JSON)
    
    # Analysis metadata
    analysis_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    startup = relationship("Startup", back_populates="analyses")
    
    def __repr__(self):
        return f"<Analysis(id='{self.id}', startup_id='{self.startup_id}', overall_score='{self.overall_score}')>"


class Report(Base):
    """Report entity model for generated PDF reports"""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    startup_id = Column(Integer, ForeignKey("startups.id"), nullable=False)
    
    # Report metadata
    report_name = Column(String(255))
    report_type = Column(String(50))
    file_path = Column(String(512))
    
    # Report generation metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    startup = relationship("Startup", back_populates="reports")
    user = relationship("User")
    
    def __repr__(self):
        return f"<Report(id='{self.id}', startup_id='{self.startup_id}', report_name='{self.report_name}')>"


class User(Base):
    """User entity model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    
    # User metadata
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_login = Column(DateTime)
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>" 