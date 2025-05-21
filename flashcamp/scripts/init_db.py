#!/usr/bin/env python
"""
Database initialization script for FlashCAMP
This script:
1. Creates the database if it doesn't exist
2. Creates the initial tables based on models in database.py
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import Base, engine
from backend.config import settings

def create_initial_migration():
    """Create an initial Alembic migration"""
    try:
        # Run the Alembic command to create a migration
        subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", "Initial migration"],
            check=True
        )
        print("✅ Created initial migration")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create migration: {e}")
        return False
    return True

def apply_migrations():
    """Apply all pending migrations"""
    try:
        # Run the Alembic command to apply migrations
        subprocess.run(
            ["alembic", "upgrade", "head"],
            check=True
        )
        print("✅ Applied migrations")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to apply migrations: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Initialize the FlashCAMP database")
    parser.add_argument("--skip-migrations", action="store_true", help="Skip running migrations")
    parser.add_argument("--create-only", action="store_true", help="Only create the database, don't run migrations")
    args = parser.parse_args()
    
    print(f"🗄️ Initializing database: {settings.DATABASE_URL or 'sqlite:///./flashcamp.db'}")
    
    # Create tables directly if skipping migrations
    if args.skip_migrations:
        Base.metadata.create_all(bind=engine)
        print("✅ Created database tables directly (skipped migrations)")
        return
    
    # Create migration
    if not args.create_only:
        if not create_initial_migration():
            return
        
        # Apply migration
        if not apply_migrations():
            return
    else:
        print("⏩ Skipping migration creation and application (--create-only flag)")
    
    print("🎉 Database initialization complete")

if __name__ == "__main__":
    main() 