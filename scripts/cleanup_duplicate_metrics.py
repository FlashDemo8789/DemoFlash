#!/usr/bin/env python3
"""
Cleanup duplicate metrics in the FLASH dataset.

This script:
1. Reads the dataset with duplicate metrics
2. Maps duplicate columns to their canonical names
3. Writes a clean version of the dataset
4. Optionally tests for column collinearity
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Tuple, Set
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
REPO_ROOT = Path(__file__).parent.parent
DATA_PATH = REPO_ROOT / "flashcamp" / "data"
SOURCE_FILE = DATA_PATH / "camp_plus_balanced_with_meta.csv" 
TARGET_FILE = DATA_PATH / "camp_plus_balanced_clean.csv"
METRICS_JSON_BACKEND = REPO_ROOT / "flashcamp" / "backend" / "contracts" / "metrics.json"
METRICS_JSON_FRONTEND = REPO_ROOT / "flashcamp" / "frontend" / "constants" / "metrics.json"

# Define duplicate mapping - canonical name -> aliases to drop
DUPLICATE_MAPPING = {
    # Advantage pillar
    "patent_count": ["patents_count"],
    "has_network_effect": ["network_effects_present"],
    
    # Market pillar
    "nps_score": ["nps"],
    "burn_rate_usd": ["monthly_burn_usd"],
    
    # Capital pillar
    "total_funding_usd": ["total_capital_raised_usd"],
    "revenue_annual_usd": ["annual_revenue_run_rate"],
    
    # People pillar
    "founders_count": ["founding_team_size"],
    "domain_expertise_years_avg": ["founder_domain_experience_years"],
    "previous_exits_count": ["prior_successful_exits_count"],
    
    # Info/Context
    "sector": ["industry"]
}

# Flatten the mapping to make it easier to use
FLAT_MAPPING = {}
for canonical, aliases in DUPLICATE_MAPPING.items():
    for alias in aliases:
        FLAT_MAPPING[alias] = canonical

def load_dataset() -> pd.DataFrame:
    """Load the original dataset with duplicates."""
    try:
        logger.info(f"Loading dataset from {SOURCE_FILE}")
        df = pd.read_csv(SOURCE_FILE)
        logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

def identify_duplicate_columns(df: pd.DataFrame, threshold: float = 0.999) -> List[Tuple[str, str, float]]:
    """
    Identify highly correlated columns that might be duplicates.
    Returns a list of tuples (col1, col2, correlation).
    """
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()
    
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find column pairs with correlation greater than threshold
    duplicates = []
    for col1 in upper.columns:
        for col2 in upper.index:
            if upper.loc[col2, col1] > threshold:
                duplicates.append((col1, col2, upper.loc[col2, col1]))
                
    return sorted(duplicates, key=lambda x: x[2], reverse=True)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by:
    1. Dropping alias columns if the canonical column exists
    2. Renaming alias columns to canonical names if the canonical doesn't exist
    """
    # Check which columns are in the dataset
    columns_in_df = set(df.columns)
    canonical_columns = set(DUPLICATE_MAPPING.keys())
    alias_columns = {alias for aliases in DUPLICATE_MAPPING.values() for alias in aliases}
    
    # Columns to drop
    to_drop = []
    # Columns to rename
    to_rename = {}
    
    for alias, canonical in FLAT_MAPPING.items():
        if alias in columns_in_df:
            if canonical in columns_in_df:
                # Both alias and canonical exist, drop the alias
                to_drop.append(alias)
                logger.info(f"Dropping column '{alias}' (duplicate of '{canonical}')")
            else:
                # Only alias exists, rename it to canonical
                to_rename[alias] = canonical
                logger.info(f"Renaming column '{alias}' to '{canonical}'")
    
    # Apply changes
    df_clean = df.copy()
    
    # Drop duplicates
    if to_drop:
        df_clean = df_clean.drop(columns=to_drop)
        logger.info(f"Dropped {len(to_drop)} duplicate columns")
    
    # Rename columns
    if to_rename:
        df_clean = df_clean.rename(columns=to_rename)
        logger.info(f"Renamed {len(to_rename)} columns to their canonical names")
    
    return df_clean

def test_for_collinearity(df: pd.DataFrame, threshold: float = 0.999) -> None:
    """Test for remaining collinear columns that might be duplicates."""
    duplicates = identify_duplicate_columns(df, threshold)
    
    if duplicates:
        logger.warning(f"Found {len(duplicates)} potentially duplicate column pairs:")
        for col1, col2, corr in duplicates:
            logger.warning(f"  {col1} <-> {col2} (correlation: {corr:.4f})")
    else:
        logger.info("No remaining duplicate columns found.")

def update_metrics_json(file_path: str) -> None:
    """Update metrics JSON file to remove duplicate metrics."""
    try:
        import json
        
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        
        # Create a set of aliases to remove
        aliases_to_remove = {alias for aliases in DUPLICATE_MAPPING.values() for alias in aliases}
        
        # Filter out aliases
        if isinstance(metrics, list):
            # Find the key field in the JSON structure
            key_field = None
            if metrics and isinstance(metrics[0], dict):
                if "key" in metrics[0]:
                    key_field = "key"
                elif "name" in metrics[0]:
                    key_field = "name"
            
            if key_field:
                metrics_filtered = [m for m in metrics if m.get(key_field) not in aliases_to_remove]
                logger.info(f"Removed {len(metrics) - len(metrics_filtered)} duplicate metrics from {file_path}")
                
                # Write back the updated metrics
                with open(file_path, 'w') as f:
                    json.dump(metrics_filtered, f, indent=2)
            else:
                logger.warning(f"Could not determine key field in {file_path}")
        else:
            logger.warning(f"Unexpected metrics format in {file_path}")
                
    except Exception as e:
        logger.error(f"Error updating metrics JSON file {file_path}: {e}")

def main():
    """Main function to clean up duplicate metrics."""
    # Load the dataset
    df = load_dataset()
    
    # Clean the dataset
    df_clean = clean_dataset(df)
    
    # Test for any remaining collinearity
    test_for_collinearity(df_clean)
    
    # Save the cleaned dataset
    try:
        logger.info(f"Saving cleaned dataset to {TARGET_FILE}")
        df_clean.to_csv(TARGET_FILE, index=False)
        logger.info(f"Saved cleaned dataset with {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error saving cleaned dataset: {e}")
        sys.exit(1)
    
    # Update metrics JSON files
    if METRICS_JSON_BACKEND.exists():
        update_metrics_json(str(METRICS_JSON_BACKEND))
    else:
        logger.warning(f"Backend metrics JSON file not found at {METRICS_JSON_BACKEND}")
    
    if METRICS_JSON_FRONTEND.exists():
        update_metrics_json(str(METRICS_JSON_FRONTEND))
    else:
        logger.warning(f"Frontend metrics JSON file not found at {METRICS_JSON_FRONTEND}")
    
    logger.info("Cleanup completed successfully!")

if __name__ == "__main__":
    main() 