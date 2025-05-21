#!/usr/bin/env python3
"""
Unit test to check for collinearity in metrics.

This test identifies columns in the dataset that are highly correlated and might be duplicates.
"""

import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

# Add the parent directory to the path so we can import from scripts
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

from scripts.cleanup_duplicate_metrics import identify_duplicate_columns

class TestMetricsCollinearity(unittest.TestCase):
    """Test cases for checking metrics collinearity."""
    
    def setUp(self):
        """Set up the test by loading the dataset."""
        self.data_path = REPO_ROOT / "flashcamp" / "data"
        self.source_file = self.data_path / "camp_plus_balanced_with_meta.csv"
        
        # If we have a clean file, use it, otherwise use the original
        self.clean_file = self.data_path / "camp_plus_balanced_clean.csv"
        if self.clean_file.exists():
            self.test_file = self.clean_file
        else:
            self.test_file = self.source_file
    
    def test_no_collinearity(self):
        """Test that there are no highly correlated columns in the dataset."""
        # Skip if file doesn't exist
        if not self.test_file.exists():
            self.skipTest(f"Dataset file {self.test_file} not found")
        
        # Load the dataset
        df = pd.read_csv(self.test_file)
        
        # Identify duplicate columns with high correlation
        duplicates = identify_duplicate_columns(df, threshold=0.999)
        
        # If we find duplicates, print them for debugging
        if duplicates:
            print("\nPotential duplicate columns found:")
            for col1, col2, corr in duplicates:
                print(f"  {col1} <-> {col2} (correlation: {corr:.4f})")
        
        # Assert that there are no duplicate columns
        self.assertEqual(len(duplicates), 0, 
                         f"Found {len(duplicates)} potentially duplicate column pairs")
    
    def test_known_duplicates_removed(self):
        """Test that known duplicate columns have been removed or renamed."""
        # Skip if clean file doesn't exist
        if not self.clean_file.exists():
            self.skipTest(f"Cleaned dataset file {self.clean_file} not found")
        
        # Known duplicates that should be removed
        known_duplicates = [
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
        
        # Load the cleaned dataset
        df = pd.read_csv(self.clean_file)
        
        # Check that none of the known duplicates are in the columns
        for duplicate in known_duplicates:
            self.assertNotIn(duplicate, df.columns, 
                            f"Known duplicate column '{duplicate}' still exists in the dataset")

if __name__ == "__main__":
    unittest.main() 