"""
Train hierarchical models for startup success prediction.
Trains four pillar models (Capital, Advantage, Market, People) and a meta-model.

Usage:
    python pipelines/train_hierarchical.py \
        --data data/gold/seed_dataset_master_final_54000_68.csv \
        --models models/v2/
"""
import argparse
import joblib
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from flashcamp.features.build_features import build_feature_vector, FEATURE_COLUMNS

# Define the four pillars for our model
PILLARS = ["capital", "advantage", "market", "people"]

# Define feature indices for each pillar
# Capital pillar (financial metrics)
CAPITAL_COLS = [
    "log_cash_on_hand", "ltv_cac_ratio", "burn_multiple", "runway_est",
    "gross_margin", "customer_concentration", "post_money_valuation"
]

# Advantage pillar (competitive advantage and product metrics)
ADVANTAGE_COLS = [
    "patent_count_norm", "network_effect", "has_data_moat", "reg_advantage",
    "tech_diff_score", "switch_cost_score", "brand_strength",
    "retention_30d", "retention_90d", "nps_score_norm"
]

# Market pillar (market size, growth and competition)
MARKET_COLS = [
    "tam_ratio", "sam_ratio", "cagr_pct", "market_growth_pct",
    "competition_intensity", "competition_hhi", "reg_level_idx"
]

# People pillar (team and leadership metrics)
PEOPLE_COLS = [
    "founders_count_norm", "team_size_norm", "domain_exp_avg",
    "prior_exits", "board_advisor_score", "team_diversity",
    "gender_div_idx", "geo_div_idx", "key_person_dependency"
]

def create_feature_masks():
    """Create masks for each pillar based on FEATURE_COLUMNS"""
    column_to_idx = {col: idx for idx, col in enumerate(FEATURE_COLUMNS)}
    
    # Define feature masks for each pillar
    masks = {
        "capital": [column_to_idx[col] for col in CAPITAL_COLS if col in column_to_idx],
        "advantage": [column_to_idx[col] for col in ADVANTAGE_COLS if col in column_to_idx],
        "market": [column_to_idx[col] for col in MARKET_COLS if col in column_to_idx],
        "people": [column_to_idx[col] for col in PEOPLE_COLS if col in column_to_idx]
    }
    
    # Validate masks - ensure no empty masks
    for pillar, mask in masks.items():
        if not mask:
            raise ValueError(f"Empty feature mask for {pillar} pillar. Check feature names.")
    
    return masks

def train_lightgbm(X, y, eval_set=None, n_trials=30, params=None):
    """
    Train a LightGBM model with Optuna hyperparameter optimization.
    
    Args:
        X: Training features
        y: Target labels
        eval_set: Optional evaluation set (X_val, y_val)
        n_trials: Number of Optuna trials
        params: Optional starting parameter set
        
    Returns:
        Trained LightGBM model and best parameters
    """
    def objective(trial):
        # Start with provided params or empty dict
        trial_params = params.copy() if params else {}
        
        # Add/override with Optuna suggested params
        trial_params.update({
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        })
        
        # Cross-validation to find best params
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            lgb_train = lgb.Dataset(X_train_cv, y_train_cv)
            lgb_val = lgb.Dataset(X_val_cv, y_val_cv, reference=lgb_train)
            
            model = lgb.train(
                trial_params,
                lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            preds = model.predict(X_val_cv)
            auc = roc_auc_score(y_val_cv, preds)
            cv_scores.append(auc)
            
        return np.mean(cv_scores)
    
    # Run Optuna hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params["objective"] = "binary"
    best_params["metric"] = "auc"
    best_params["verbosity"] = -1
    
    lgb_train = lgb.Dataset(X, y)
    
    if eval_set:
        X_val, y_val = eval_set
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        model = lgb.train(
            best_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
    else:
        model = lgb.train(best_params, lgb_train, num_boost_round=500)
    
    return model, best_params

def train_xgboost(X, y, eval_set=None, n_trials=30, params=None):
    """
    Train an XGBoost model with Optuna hyperparameter optimization.
    
    Args:
        X: Training features
        y: Target labels
        eval_set: Optional evaluation set (X_val, y_val)
        n_trials: Number of Optuna trials
        params: Optional starting parameter set
        
    Returns:
        Trained XGBoost model and best parameters
    """
    def objective(trial):
        # Start with provided params or empty dict
        trial_params = params.copy() if params else {}
        
        # Add/override with Optuna suggested params
        trial_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'verbosity': 0,
            'random_state': 42
        })
        
        # Cross-validation to find best params
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(**trial_params)
            model.fit(
                X_train_cv, y_train_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            preds = model.predict_proba(X_val_cv)[:, 1]
            auc = roc_auc_score(y_val_cv, preds)
            cv_scores.append(auc)
            
        return np.mean(cv_scores)
    
    # Run Optuna hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "auc"
    best_params["verbosity"] = 0
    best_params["random_state"] = 42
    
    model = xgb.XGBClassifier(**best_params)
    
    if eval_set:
        X_val, y_val = eval_set
        model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50
        )
    else:
        model.fit(X, y)
    
    return model, best_params

def generate_shap_plots(model, X, feature_names, output_dir, model_name):
    """Generate and save SHAP visualizations for model explanations"""
    # Create explainer
    if isinstance(model, lgb.Booster):
        explainer = shap.TreeExplainer(model)
    else:  # XGBoost model
        explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # For XGBoost binary classification, we may need to extract the relevant values
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]  # Use class 1 for binary classification
    
    # Create output directory
    output_dir = Path(output_dir) / "shap_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Generate bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return shap_values

def evaluate_model(model, X, y, model_name):
    """Evaluate model performance and return metrics"""
    # Make predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X)
    
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y, y_prob)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    
    # Calculate calibration error
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    
    # Print metrics
    print(f"\n--- {model_name} Performance Metrics ---")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Calibration Error: {calibration_error:.4f}")
    
    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "calibration_error": float(calibration_error)
    }

def train():
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to the dataset CSV file")
    ap.add_argument("--models", required=True, help="Directory to save trained models")
    ap.add_argument("--sample", type=int, default=0, help="Sample n rows for quick testing (0 = all data)")
    ap.add_argument("--trials", type=int, default=30, help="Number of Optuna trials for hyperparameter search")
    args = ap.parse_args()

    # Create model directory
    model_dir = Path(args.models)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create reports directory for SHAP plots
    reports_dir = Path(__file__).resolve().parents[2] / "reports" / "assets"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {args.data}...")
    # Load the dataset (support both CSV and parquet)
    if args.data.endswith('.csv'):
        df = pd.read_csv(args.data)
    else:
        df = pd.read_parquet(args.data)
    
    # Sample data if requested
    if args.sample > 0:
        df = df.sample(min(args.sample, len(df)), random_state=42)
    
    print(f"Dataset loaded with {len(df)} samples.")
    
    # Convert success_label from string to numeric (pass=1, fail=0)
    if "success_label" in df.columns:
        if df["success_label"].dtype == 'object':
            df["success_label_numeric"] = df["success_label"].map({"pass": 1, "fail": 0})
        else:
            df["success_label_numeric"] = df["success_label"]
    else:
        raise ValueError("Dataset must contain 'success_label' column")
    
    # Extract success rate stats
    success_rate = df["success_label_numeric"].mean()
    print(f"Success rate in dataset: {success_rate:.2%}")
    
    # Process each row to extract features
    print("Extracting features...")
    X_list = []
    for _, row in df.iterrows():
        try:
            X_list.append(build_feature_vector(row.to_dict()))
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    # Stack the features together
    X_all = np.vstack([x for x in X_list])
    y_all = df["success_label_numeric"].values
    
    print(f"Feature matrix shape: {X_all.shape}")
    
    # Create feature masks for each pillar
    feature_masks = create_feature_masks()
    
    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train pillar models
    print("\n=== Training Pillar Models ===")
    
    pillar_models = {}
    pillar_params = {}
    
    # Store pillar predictions for meta-model training
    train_pillar_scores = np.zeros((X_train.shape[0], len(PILLARS)))
    val_pillar_scores = np.zeros((X_val.shape[0], len(PILLARS)))
    test_pillar_scores = np.zeros((X_test.shape[0], len(PILLARS)))
    
    for i, pillar in enumerate(PILLARS):
        print(f"\n--- Training {pillar.upper()} Pillar Model ---")
        mask = feature_masks[pillar]
        
        # Get feature names for this pillar
        pillar_features = [FEATURE_COLUMNS[idx] for idx in mask]
        print(f"Using {len(pillar_features)} features for {pillar} pillar")
        
        # Train model with Optuna
        model, params = train_lightgbm(
            X_train[:, mask], y_train,
            eval_set=(X_val[:, mask], y_val),
            n_trials=args.trials
        )
        
        # Save model
        joblib.dump(model, model_dir / f"{pillar}_lgbm.joblib")
        
        # Store model and parameters
        pillar_models[pillar] = model
        pillar_params[pillar] = params
        
        # Get pillar scores for meta-model
        train_pillar_scores[:, i] = model.predict(X_train[:, mask])
        val_pillar_scores[:, i] = model.predict(X_val[:, mask])
        test_pillar_scores[:, i] = model.predict(X_test[:, mask])
        
        # Evaluate on validation set
        metrics = evaluate_model(model, X_val[:, mask], y_val, f"{pillar} pillar")
        pillar_params[pillar]["metrics"] = metrics
        
        # Generate SHAP plots
        generate_shap_plots(
            model, 
            X_val[:, mask], 
            pillar_features, 
            reports_dir, 
            f"{pillar}_pillar"
        )
    
    # Train meta-model
    print("\n=== Training Meta-Model ===")
    meta_model, meta_params = train_xgboost(
        train_pillar_scores, y_train,
        eval_set=(val_pillar_scores, y_val),
        n_trials=args.trials
    )
    
    # Save the meta-model
    joblib.dump(meta_model, model_dir / "success_xgb.joblib")
    
    # Evaluate meta-model
    meta_metrics = evaluate_model(meta_model, test_pillar_scores, y_test, "Meta-model")
    meta_params["metrics"] = meta_metrics
    
    # Generate SHAP plots for meta-model
    generate_shap_plots(
        meta_model, 
        val_pillar_scores, 
        PILLARS, 
        reports_dir, 
        "meta_model"
    )
    
    # Save metadata about the models
    metadata = {
        "dataset": args.data,
        "dataset_size": len(df),
        "success_rate": float(success_rate),
        "feature_count": X_all.shape[1],
        "pillar_models": {
            pillar: {
                "feature_count": len(feature_masks[pillar]),
                "features": [FEATURE_COLUMNS[idx] for idx in feature_masks[pillar]],
                "params": pillar_params.get(pillar, {})
            }
            for pillar in PILLARS
        },
        "meta_model": {
            "params": meta_params,
            "metrics": meta_metrics
        },
        "training_date": pd.Timestamp.now().isoformat()
    }
    
    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModels and metadata saved to {model_dir}")
    print(f"SHAP visualizations saved to {reports_dir / 'shap_plots'}")
    
    # Print final performance summary
    print("\n=== Performance Summary ===")
    for pillar in PILLARS:
        metrics = pillar_params[pillar]["metrics"]
        print(f"{pillar.upper()} Pillar: AUC = {metrics['auc']:.4f}, Calibration Error = {metrics['calibration_error']:.4f}")
    
    print(f"META-MODEL: AUC = {meta_metrics['auc']:.4f}, Calibration Error = {meta_metrics['calibration_error']:.4f}")
    
if __name__ == "__main__":
    train() 