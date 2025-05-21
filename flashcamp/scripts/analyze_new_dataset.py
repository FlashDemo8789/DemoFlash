#!/usr/bin/env python3
"""
Dataset Analysis Script

This script analyzes the structure of the new dataset to understand its columns,
data types, and statistical properties. This information will help us update
the metrics contract and feature engineering code.

Usage:
    python flashcamp/scripts/analyze_new_dataset.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = Path("flashcamp/data")
NEW_DATASET = DATA_DIR / "seed_dataset_master_final_54000_68.csv"
METRICS_CONTRACT = Path("flashcamp/contracts/metrics.json")
OUTPUT_DIR = Path("flashcamp/reports")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_dataset(file_path: Path, nrows=None):
    """Load a dataset with error handling"""
    try:
        if nrows:
            logger.info(f"Loading first {nrows} rows from {file_path}")
            df = pd.read_csv(file_path, nrows=nrows)
        else:
            logger.info(f"Loading dataset from {file_path}")
            # First check the number of columns
            df_header = pd.read_csv(file_path, nrows=0)
            if len(df_header.columns) > 50:
                logger.info(f"Large dataset detected with {len(df_header.columns)} columns, sampling 5000 rows")
                df = pd.read_csv(file_path, nrows=5000)
            else:
                df = pd.read_csv(file_path)
            
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        sys.exit(1)

def analyze_dataset(df):
    """Analyze a dataset and return statistics"""
    logger.info("Analyzing dataset structure...")
    
    # Basic info
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Column analysis
    column_analysis = []
    for col in df.columns:
        col_info = {
            "name": col,
            "type": str(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
            "missing_percent": float(df[col].isnull().mean() * 100),
        }
        
        # Add numeric stats if applicable
        if np.issubdtype(df[col].dtype, np.number):
            col_info.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "unique_values": int(df[col].nunique())
            })
        # Add string stats if applicable
        elif df[col].dtype == 'object':
            col_info["unique_values"] = int(df[col].nunique())
            # Sample some values if not too many unique values
            if df[col].nunique() < 20:
                col_info["unique_samples"] = df[col].dropna().unique().tolist()[:10]  # Limit to 10 samples
        
        # Special handling for boolean columns
        elif df[col].dtype == 'bool':
            col_info["true_count"] = int(df[col].sum())
            col_info["true_percent"] = float(df[col].mean() * 100)
        
        column_analysis.append(col_info)
    
    # Check for 'success_label' column which is crucial for model training
    if 'success_label' in df.columns:
        success_info = {
            "success_rate": float(df['success_label'].mean()) if df['success_label'].dtype != 'object' else None,
            "distribution": df['success_label'].value_counts().to_dict() if df['success_label'].nunique() < 10 else None
        }
        logger.info(f"Success label information: {success_info}")
    else:
        logger.warning("No 'success_label' column found in the dataset")
        success_info = {"warning": "success_label column not found"}
    
    return {
        "basic_info": info,
        "column_analysis": column_analysis,
        "success_label": success_info
    }

def generate_metrics_contract(column_analysis):
    """Generate a draft metrics contract based on column analysis"""
    metrics_contract = {}
    
    for col in column_analysis:
        col_name = col["name"]
        col_type = col["type"]
        
        # Map pandas dtype to contract type
        if 'int' in col_type or 'float' in col_type:
            contract_type = "number"
        elif col_type == 'bool':
            contract_type = "boolean"
        elif 'datetime' in col_type:
            contract_type = "string"  # Store dates as strings in the contract
        else:
            contract_type = "string"
        
        # Check if it might be an array based on unique values
        if col_type == 'object' and col.get("unique_samples") and any(isinstance(x, list) for x in col.get("unique_samples", [])):
            contract_type = "array"
        
        # Determine if field should be required (heuristic: low missing rate)
        required = col["missing_percent"] < 10
        
        # Create a description based on stats
        description = f"{col_name} - "
        if contract_type == "number":
            description += f"Range: {col.get('min', 'N/A')} to {col.get('max', 'N/A')}"
        elif "unique_values" in col:
            description += f"Has {col['unique_values']} unique values"
        
        metrics_contract[col_name] = {
            "type": contract_type,
            "required": required,
            "description": description
        }
    
    return metrics_contract

def write_report(analysis, output_path):
    """Write a detailed analysis report"""
    logger.info(f"Writing analysis report to {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("# Dataset Analysis Report\n\n")
        
        # Basic information
        f.write("## Basic Information\n\n")
        f.write(f"- Rows: {analysis['basic_info']['rows']}\n")
        f.write(f"- Columns: {analysis['basic_info']['columns']}\n")
        f.write(f"- Missing Values: {analysis['basic_info']['missing_values']}\n")
        f.write(f"- Memory Usage: {analysis['basic_info']['memory_usage_mb']:.2f} MB\n\n")
        
        # Success label information
        f.write("## Success Label Information\n\n")
        success_info = analysis['success_label']
        if "warning" in success_info:
            f.write(f"- Warning: {success_info['warning']}\n\n")
        else:
            f.write(f"- Success Rate: {success_info.get('success_rate', 'N/A')}\n")
            if success_info.get('distribution'):
                f.write("- Distribution:\n")
                for key, value in success_info['distribution'].items():
                    f.write(f"  - {key}: {value}\n")
            f.write("\n")
        
        # Column analysis
        f.write("## Column Analysis\n\n")
        f.write("| Column Name | Type | Missing (%) | Unique Values | Notes |\n")
        f.write("|------------|------|-------------|---------------|-------|\n")
        
        for col in analysis['column_analysis']:
            name = col['name']
            dtype = col['type']
            missing = f"{col['missing_percent']:.1f}%"
            unique = str(col.get('unique_values', 'N/A'))
            
            notes = []
            # Add min/max for numeric columns
            if 'min' in col and 'max' in col:
                notes.append(f"Range: {col['min']} - {col['max']}")
            
            # Add sample values for categorical columns with few unique values
            if 'unique_samples' in col:
                samples = str(col['unique_samples']).replace('|', '\\|')  # Escape pipe char for markdown tables
                notes.append(f"Samples: {samples}")
            
            # Add true percentage for boolean columns
            if 'true_percent' in col:
                notes.append(f"True: {col['true_percent']:.1f}%")
            
            f.write(f"| {name} | {dtype} | {missing} | {unique} | {'; '.join(notes)} |\n")

def main():
    logger.info("Starting dataset analysis...")
    
    # Check if dataset exists
    if not NEW_DATASET.exists():
        logger.error(f"Dataset not found at {NEW_DATASET}")
        return
    
    # Load dataset
    df = load_dataset(NEW_DATASET)
    
    # Analyze dataset
    analysis = analyze_dataset(df)
    
    # Write report
    report_path = OUTPUT_DIR / "dataset_analysis_report.md"
    write_report(analysis, report_path)
    
    # Generate draft metrics contract
    metrics_contract = generate_metrics_contract(analysis["column_analysis"])
    metrics_contract_path = OUTPUT_DIR / "draft_metrics_contract.json"
    with open(metrics_contract_path, 'w') as f:
        json.dump(metrics_contract, f, indent=2)
    
    logger.info(f"Analysis complete! Report saved to {report_path}")
    logger.info(f"Draft metrics contract saved to {metrics_contract_path}")
    
    print("\nNext steps:")
    print(f"1. Review the report at {report_path}")
    print(f"2. Review and refine the draft metrics contract at {metrics_contract_path}")
    print("3. Update the official metrics contract in flashcamp/contracts/metrics.json")
    print("4. Run the schema generator to update the codebase")

if __name__ == "__main__":
    main() 