#!/usr/bin/env python3
"""
Verify that metrics cleanup has been properly applied throughout the system.

This script:
1. Checks that duplicate metrics have been removed from the dataset
2. Verifies that metrics.json files in both frontend and backend are clean
3. Confirms that Pydantic schemas have been updated
4. Checks TypeScript types for any references to duplicates
"""

import json
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
REPO_ROOT = Path(__file__).parent.parent
DATA_PATH = REPO_ROOT / "flashcamp" / "data"
CLEAN_DATA_FILE = DATA_PATH / "camp_plus_balanced_clean.csv"
BACKEND_METRICS_JSON = REPO_ROOT / "flashcamp" / "backend" / "contracts" / "metrics.json"
FRONTEND_METRICS_JSON = REPO_ROOT / "flashcamp" / "frontend" / "constants" / "metrics.json"
BACKEND_SCHEMA_PY = REPO_ROOT / "flashcamp" / "backend" / "app" / "schemas.py"
FRONTEND_TYPES_TS = REPO_ROOT / "flashcamp" / "frontend" / "types" / "metrics.ts"

# Define duplicate metrics we should have removed
DUPLICATE_METRICS = [
    "patents_count",
    "network_effects_present",
    "nps",
    "monthly_burn_usd",
    "total_capital_raised_usd",
    "annual_revenue_run_rate",
    "founding_team_size",
    "founder_domain_experience_years",
    "prior_successful_exits_count",
    "industry"
]

def check_csv_data():
    """Check that the clean CSV data file exists and doesn't contain duplicates."""
    if not CLEAN_DATA_FILE.exists():
        logger.error(f"Clean data file not found at {CLEAN_DATA_FILE}")
        return False
    
    try:
        # Try to read the first line to get headers
        with open(CLEAN_DATA_FILE, 'r') as f:
            header_line = f.readline().strip()
            headers = header_line.split(',')
        
        # Check for duplicates in headers
        for duplicate in DUPLICATE_METRICS:
            if duplicate in headers:
                logger.error(f"Duplicate metric '{duplicate}' found in clean CSV header")
                return False
        
        logger.info("✅ CSV data is clean (no duplicate metrics found in header)")
        return True
    except Exception as e:
        logger.error(f"Error checking CSV data: {e}")
        return False

def check_json_file(file_path, field_name="key"):
    """Check that a JSON file doesn't contain duplicate metrics."""
    if not file_path.exists():
        logger.error(f"JSON file not found at {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check data format (list of objects)
        if not isinstance(data, list):
            logger.error(f"Unexpected format in {file_path} (expected list)")
            return False
        
        # Extract metric keys/names
        metrics = [item.get(field_name) for item in data if field_name in item]
        
        # Check for duplicates
        for duplicate in DUPLICATE_METRICS:
            if duplicate in metrics:
                logger.error(f"Duplicate metric '{duplicate}' found in {file_path}")
                return False
        
        logger.info(f"✅ JSON file {file_path.name} is clean (no duplicate metrics found)")
        return True
    except Exception as e:
        logger.error(f"Error checking JSON file {file_path}: {e}")
        return False

def check_schema_file():
    """Check that the Pydantic schema doesn't contain duplicate metrics."""
    if not BACKEND_SCHEMA_PY.exists():
        logger.error(f"Backend schema file not found at {BACKEND_SCHEMA_PY}")
        return False
    
    try:
        with open(BACKEND_SCHEMA_PY, 'r') as f:
            content = f.read()
        
        # Simple check for field declarations with duplicate names
        for duplicate in DUPLICATE_METRICS:
            if f"{duplicate}:" in content:
                logger.error(f"Duplicate metric '{duplicate}' found in backend schema")
                return False
        
        logger.info("✅ Backend schema is clean (no duplicate metrics found)")
        return True
    except Exception as e:
        logger.error(f"Error checking backend schema: {e}")
        return False

def check_typescript_types():
    """Check that the TypeScript types file doesn't contain duplicate metrics."""
    if not FRONTEND_TYPES_TS.exists():
        logger.error(f"Frontend types file not found at {FRONTEND_TYPES_TS}")
        return False
    
    try:
        with open(FRONTEND_TYPES_TS, 'r') as f:
            content = f.read()
        
        # Simple check for field declarations with duplicate names
        for duplicate in DUPLICATE_METRICS:
            if f"{duplicate}?" in content:
                logger.error(f"Duplicate metric '{duplicate}' found in frontend types")
                return False
        
        logger.info("✅ Frontend types are clean (no duplicate metrics found)")
        return True
    except Exception as e:
        logger.error(f"Error checking frontend types: {e}")
        return False

def main():
    """Main function to verify metrics cleanup."""
    logger.info("Verifying metrics cleanup...")
    
    # Check CSV data
    csv_clean = check_csv_data()
    
    # Check JSON files
    backend_json_clean = check_json_file(BACKEND_METRICS_JSON)
    frontend_json_clean = check_json_file(FRONTEND_METRICS_JSON, field_name="name")
    
    # Check schema files
    backend_schema_clean = check_schema_file()
    frontend_types_clean = check_typescript_types()
    
    # Overall result
    all_clean = csv_clean and backend_json_clean and frontend_json_clean and backend_schema_clean and frontend_types_clean
    
    if all_clean:
        logger.info("✅ ALL CHECKS PASSED: Metrics cleanup has been successfully applied throughout the system")
    else:
        logger.error("❌ CHECKS FAILED: Metrics cleanup has not been fully applied")
        logger.error("Please run 'python scripts/cleanup_duplicate_metrics.py' and then 'python scripts/gen_metrics.py'")
        sys.exit(1)

if __name__ == "__main__":
    main() 