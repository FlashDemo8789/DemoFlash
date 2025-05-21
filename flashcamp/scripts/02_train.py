#!/usr/bin/env python
"""
02_train.py
───────────
Retrains:
• Four pillar regression models (LightGBM) – optional future use
• Composite success XGBoost classifier  – saved to models/success_xgb.joblib

Usage:
    python flashcamp/scripts/02_train.py --data data/gold/v1.parquet --models flashcamp/models
"""

import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
import json # Import json
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1] # flashcamp/

# Read metrics from the correct frontend path
METRICS_PATH = ROOT / "frontend/src/constants/metrics.json"
if not METRICS_PATH.is_file():
    # Fallback to older path if src/constants doesn't exist yet
    METRICS_PATH = ROOT / "frontend/constants/metrics.json"
    if not METRICS_PATH.is_file():
      print(f"Error: Cannot find metrics.json at {ROOT / 'frontend/src/constants/metrics.json'} or {ROOT / 'frontend/constants/metrics.json'}")
      import sys; sys.exit(1) # Add import sys

METRICS_ALL = json.loads(METRICS_PATH.read_text())

# Filter metrics to include only numeric and boolean types for XGBoost
METRICS_FOR_XGB = [m["name"] for m in METRICS_ALL if m["type"] in ["number", "checkbox"]]

PILLARS = ["capital", "advantage", "market", "people"]

def train_pillar(df, target, out_dir: Path):
    # Pillar training might use all metrics or a specific subset - adjust if needed
    # For now, assume it also uses numeric/checkbox for consistency, 
    # but this depends on how these optional models are intended to be used.
    X = df[METRICS_FOR_XGB]
    y = df[target]
    model = LGBMRegressor(n_estimators=400, num_leaves=64)
    model.fit(X, y)
    joblib.dump(model, out_dir / f"{target}_lgbm.joblib")

def train_success(df, out_dir: Path):
    # Use only the filtered numeric/boolean features for training
    X = df[METRICS_FOR_XGB]
    # Ensure 'success_label' exists in the DataFrame
    if 'success_label' not in df.columns:
        print("Error: 'success_label' column not found in the input data.")
        import sys; sys.exit(1)
    y = df["success_label"]

    # Handle potential NaN or infinite values in features or labels
    # A simple approach is to fill NaNs, e.g., with median or mean
    # More robust handling might involve imputation strategies
    X = X.fillna(X.median(numeric_only=True))
    # Check for infinite values if necessary
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median(numeric_only=True)) # Fill again if infs were replaced
    
    # Ensure label 'y' is clean (e.g., no NaNs if it's the target)
    if y.isnull().any():
       print("Warning: NaNs found in 'success_label'. Attempting to drop rows.")
       # Drop rows where label is NaN
       valid_indices = y.dropna().index
       X = X.loc[valid_indices]
       y = y.loc[valid_indices]
       if X.empty:
           print("Error: No valid data remaining after dropping NaN labels.")
           import sys; sys.exit(1)
           
    # Convert target to integers if it's not already (required by XGBoost)
    y = y.astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Ensure y_train and y_val are also integers
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    
    clf = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=4,
        eval_metric="auc",
        # Remove deprecated parameter for XGBoost >= 1.6.0
        # use_label_encoder=False 
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    print(f"✅ success_xgb joblib trained – AUC {auc:0.3f}")
    joblib.dump(clf, out_dir / "success_xgb.joblib")

def main(data_path: Path, model_dir: Path):
    df = pd.read_parquet(data_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. pillar models (optional)
    # for pillar in PILLARS:
    #     out_col = f"{pillar}_score"
    #     if out_col in df:
    #         train_pillar(df, out_col, model_dir)

    # 2. success model
    train_success(df, model_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--models", required=True)
    args = p.parse_args()
    main(Path(args.data), Path(args.models)) 