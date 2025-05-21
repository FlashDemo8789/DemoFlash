#!/usr/bin/env python3
"""
Combined script to clean up duplicate metrics and verify the cleanup.

This script:
1. Runs cleanup_duplicate_metrics.py to clean up duplicate metrics
2. Runs gen_metrics.py to update schema files
3. Runs verify_metrics_cleanup.py to verify the cleanup
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
REPO_ROOT = Path(__file__).parent.parent
CLEANUP_SCRIPT = REPO_ROOT / "scripts" / "cleanup_duplicate_metrics.py"
GEN_METRICS_SCRIPT = REPO_ROOT / "scripts" / "gen_metrics.py"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_metrics_cleanup.py"

def run_script(script_path):
    """Run a Python script and return success/failure."""
    try:
        logger.info(f"Running {script_path.name}...")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logger.info(f"✅ Successfully ran {script_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error running {script_path.name}")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Output: {e.output}")
        logger.error(f"Error: {e.stderr}")
        return False

def main():
    """Main function to clean up and verify metrics."""
    logger.info("Starting metrics cleanup and verification...")
    
    # Check if scripts exist
    for script in [CLEANUP_SCRIPT, GEN_METRICS_SCRIPT, VERIFY_SCRIPT]:
        if not script.exists():
            logger.error(f"Script not found: {script}")
            sys.exit(1)
    
    # Run cleanup script
    if not run_script(CLEANUP_SCRIPT):
        logger.error("Cleanup script failed. Exiting.")
        sys.exit(1)
    
    # Run gen_metrics script
    if not run_script(GEN_METRICS_SCRIPT):
        logger.error("Schema generation script failed. Exiting.")
        sys.exit(1)
    
    # Run verify script
    if not run_script(VERIFY_SCRIPT):
        logger.error("Verification script failed. Exiting.")
        sys.exit(1)
    
    logger.info("✅ SUCCESS: Metrics cleanup and verification completed successfully")
    logger.info("All duplicate metrics have been removed from the dataset and schema files")

if __name__ == "__main__":
    main() 