"""
Train pillar models + composite success model in one shot.
Usage:
    python pipelines/train_baseline.py \
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
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from flashcamp.features.build_features import build_feature_vector, FEATURE_COLUMNS

# Define the four pillars for our model
PILLARS = ["capital", "advantage", "market", "people"]

def train_lightgbm(X, y, eval_set=None):
    """
    Train a LightGBM model with Optuna hyperparameter optimization.
    
    Args:
        X: Training features
        y: Target labels
        eval_set: Optional evaluation set (X_val, y_val)
        
    Returns:
        Trained LightGBM model
    """
    def objective(trial):
        params = dict(
            objective="binary",
            metric="auc",
            verbosity=-1,
            learning_rate=trial.suggest_float("lr", 0.01, 0.3),
            num_leaves=trial.suggest_int("num_leaves", 20, 150),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 100),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.4, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
        )
        
        # Cross-validation to find best params
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            lgb_train = lgb.Dataset(X_train_cv, y_train_cv)
            lgb_val = lgb.Dataset(X_val_cv, y_val_cv, reference=lgb_train)
            
            model = lgb.train(
                params,
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
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
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

def train_xgboost(X, y, eval_set=None):
    """
    Train an XGBoost model with Optuna hyperparameter optimization.
    
    Args:
        X: Training features
        y: Target labels
        eval_set: Optional evaluation set (X_val, y_val)
        
    Returns:
        Trained XGBoost model
    """
    def objective(trial):
        params = {
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
        }
        
        # Cross-validation to find best params
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(**params)
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
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
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

def train():
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to the dataset CSV file")
    ap.add_argument("--models", required=True, help="Directory to save trained models")
    ap.add_argument("--sample", type=int, default=0, help="Sample n rows for quick testing (0 = all data)")
    args = ap.parse_args()

    # Create model directory
    model_dir = Path(args.models)
    model_dir.mkdir(parents=True, exist_ok=True)
    
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
    if df["success_label"].dtype == 'object':
        df["success_label_numeric"] = df["success_label"].map({"pass": 1, "fail": 0})
    else:
        df["success_label_numeric"] = df["success_label"]
    
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
    
    # Define feature indices for each pillar
    # Capital pillar (financial metrics)
    capital_cols = [
        "log_cash_on_hand", "ltv_cac_ratio", "burn_multiple", "runway_est",
        "gross_margin", "customer_concentration", "post_money_valuation"
    ]
    
    # Advantage pillar (competitive advantage and product metrics)
    advantage_cols = [
        "patent_count_norm", "network_effect", "has_data_moat", "reg_advantage",
        "tech_diff_score", "switch_cost_score", "brand_strength",
        "retention_30d", "retention_90d", "nps_score_norm"
    ]
    
    # Market pillar (market size, growth and competition)
    market_cols = [
        "tam_ratio", "sam_ratio", "cagr_pct", "market_growth_pct",
        "competition_intensity", "competition_hhi", "reg_level_idx"
    ]
    
    # People pillar (team and leadership metrics)
    people_cols = [
        "founders_count_norm", "team_size_norm", "domain_exp_avg",
        "prior_exits", "board_advisor_score", "team_diversity",
        "gender_div_idx", "geo_div_idx", "key_person_dependency"
    ]
    
    # Performance pillar (operational metrics)
    performance_cols = [
        "conversion_rate", "user_growth_rate", "customer_count_norm",
        "scalability_score", "net_dollar_retention"
    ]
    
    # Create mapping of column names to indices
    column_to_idx = {col: idx for idx, col in enumerate(FEATURE_COLUMNS)}
    
    # Define feature masks for each pillar
    masks = {
        "capital": [column_to_idx[col] for col in capital_cols if col in column_to_idx],
        "advantage": [column_to_idx[col] for col in advantage_cols if col in column_to_idx],
        "market": [column_to_idx[col] for col in market_cols if col in column_to_idx],
        "people": [column_to_idx[col] for col in people_cols if col in column_to_idx],
        "performance": [column_to_idx[col] for col in performance_cols if col in column_to_idx]
    }
    
    # Split data into train and evaluation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # Store pillar scores for training the composite model
    train_pillar_scores = np.zeros((len(X_train), len(PILLARS)))
    val_pillar_scores = np.zeros((len(X_val), len(PILLARS)))
    
    # Train pillar models
    pillar_params = {}
    
    for i, pillar in enumerate(PILLARS):
        print(f"\nTraining {pillar} pillar model...")
        mask = masks[pillar]
        
        # Train LightGBM model for this pillar
        model, params = train_lightgbm(
            X_train[:, mask], y_train, 
            eval_set=(X_val[:, mask], y_val)
        )
        
        # Save the model
        joblib.dump(model, model_dir / f"{pillar}_lgbm.joblib")
        
        # Generate predictions for the composite model
        train_pillar_scores[:, i] = model.predict(X_train[:, mask])
        val_pillar_scores[:, i] = model.predict(X_val[:, mask])
        
        # Calculate AUC for this pillar
        pillar_auc = roc_auc_score(y_val, val_pillar_scores[:, i])
        print(f"{pillar.title()} pillar AUC: {pillar_auc:.4f}")
        
        # Store parameters
        pillar_params[pillar] = params
    
    # Train performance model separately (not part of the composite model)
    if "performance" in masks:
        print("\nTraining performance pillar model...")
        mask = masks["performance"]
        perf_model, perf_params = train_lightgbm(
            X_train[:, mask], y_train,
            eval_set=(X_val[:, mask], y_val)
        )
        joblib.dump(perf_model, model_dir / "performance_lgbm.joblib")
        perf_preds = perf_model.predict(X_val[:, mask])
        perf_auc = roc_auc_score(y_val, perf_preds)
        print(f"Performance pillar AUC: {perf_auc:.4f}")
        pillar_params["performance"] = perf_params
    
    # Train composite success model
    print("\nTraining composite success model...")
    composite_model, composite_params = train_xgboost(
        train_pillar_scores, y_train,
        eval_set=(val_pillar_scores, y_val)
    )
    
    # Save the composite model
    joblib.dump(composite_model, model_dir / "success_xgb.joblib")
    
    # Evaluate composite model
    val_preds = composite_model.predict_proba(val_pillar_scores)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    val_acc = accuracy_score(y_val, val_preds > 0.5)
    
    print(f"\nComposite success model AUROC: {val_auc:.4f}")
    print(f"Composite success model accuracy: {val_acc:.4f}")
    
    # Save metadata about the models
    metadata = {
        "dataset": args.data,
        "dataset_size": len(df),
        "success_rate": float(success_rate),
        "feature_count": X_all.shape[1],
        "pillar_metrics": {
            pillar: {
                "feature_count": len(masks[pillar]),
                "features": [FEATURE_COLUMNS[idx] for idx in masks[pillar]],
                "params": pillar_params.get(pillar, {})
            }
            for pillar in masks.keys()
        },
        "composite_metrics": {
            "auc": float(val_auc),
            "accuracy": float(val_acc),
            "params": composite_params
        },
        "training_date": pd.Timestamp.now().isoformat()
    }
    
    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModels and metadata saved to {model_dir}")

if __name__ == "__main__":
    train() 