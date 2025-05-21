"""
Generate SHAP waterfall PNGs and HTML files for every startup in a dataset.
Save to reports/assets/<startup_id>.png and reports/assets/<startup_id>.html

Usage:
    python pipelines/gen_shap.py \
        --data data/gold/seed_dataset_master_final_54000_68.csv \
        --models models/v2/ \
        --out reports/assets/shap_v2/ \
        --sample 100
"""
import argparse
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from flashcamp.features.build_features import build_feature_vector, FEATURE_COLUMNS
from flashcamp.pipelines.train_baseline import PILLARS

def generate_shap_for_model(model, X, feature_names, out_dir, filename_prefix, max_display=10):
    """Generate SHAP plots for a specific model"""
    # Create Tree explainer
    explainer = shap.TreeExplainer(model)
    
    # Generate SHAP values
    shap_values = explainer(X)
    
    # Generate waterfall plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False, max_display=max_display)
    
    # Save PNG
    plt.savefig(out_dir / f"{filename_prefix}.png", bbox_inches="tight", dpi=150)
    plt.close()
    
    # Save HTML
    shap.save_html(str(out_dir / f"{filename_prefix}.html"), 
                   shap.plots.waterfall(shap_values[0], show=False, max_display=max_display))
    
    # Return SHAP values for further analysis
    return shap_values

def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to the dataset CSV or parquet file")
    ap.add_argument("--models", required=True, help="Directory containing the trained models")
    ap.add_argument("--out", required=True, help="Output directory for SHAP plots")
    ap.add_argument("--sample", type=int, default=0, help="Generate plots for a sample of startups")
    args = ap.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    if args.data.endswith('.csv'):
        data = pd.read_csv(args.data)
    else:
        data = pd.read_parquet(args.data)
    
    # Sample if requested
    if args.sample > 0:
        data = data.sample(min(args.sample, len(data)), random_state=42)
    
    print(f"Generating SHAP values for {len(data)} startups...")
    
    # Set up output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    model_dir = Path(args.models)
    
    # Load composite model
    composite_model = joblib.load(model_dir / "success_xgb.joblib")
    
    # Load pillar models
    pillar_models = {}
    for pillar in PILLARS:
        try:
            pillar_models[pillar] = joblib.load(model_dir / f"{pillar}_lgbm.joblib")
            print(f"Loaded {pillar} model")
        except FileNotFoundError:
            print(f"Warning: {pillar} model not found")
    
    # Load feature mapping for pillar masks
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
    
    # Create mapping of column names to indices
    column_to_idx = {col: idx for idx, col in enumerate(FEATURE_COLUMNS)}
    
    # Define feature masks for each pillar
    masks = {
        "capital": [column_to_idx[col] for col in capital_cols if col in column_to_idx],
        "advantage": [column_to_idx[col] for col in advantage_cols if col in column_to_idx],
        "market": [column_to_idx[col] for col in market_cols if col in column_to_idx],
        "people": [column_to_idx[col] for col in people_cols if col in column_to_idx]
    }
    
    # Process each startup
    for i, (_, row) in enumerate(tqdm(data.iterrows(), total=len(data))):
        try:
            startup_id = row['startup_id']
            startup_name = row.get('startup_name', f"Startup_{startup_id}")
            
            # Create feature vector
            X = build_feature_vector(row.to_dict())
            
            # Generate pillar predictions (required for composite model)
            pillar_scores = np.zeros((1, len(PILLARS)))
            
            for j, pillar in enumerate(PILLARS):
                if pillar in pillar_models:
                    # Get the mask for this pillar
                    mask = masks[pillar]
                    # Predict using the pillar model
                    pillar_scores[0, j] = pillar_models[pillar].predict(X[:, mask])
            
            # Generate composite model SHAP values
            startup_dir = out_dir / startup_id
            startup_dir.mkdir(exist_ok=True)
            
            # Generate SHAP for composite model
            composite_shap = generate_shap_for_model(
                composite_model, 
                pillar_scores, 
                PILLARS,
                startup_dir, 
                "composite"
            )
            
            # Generate SHAP for individual pillar models
            for pillar in PILLARS:
                if pillar in pillar_models:
                    mask = masks[pillar]
                    feature_names = [FEATURE_COLUMNS[idx] for idx in mask]
                    pillar_shap = generate_shap_for_model(
                        pillar_models[pillar],
                        X[:, mask],
                        feature_names,
                        startup_dir,
                        pillar
                    )
            
            # Save metadata about this startup's SHAP values
            summary = {
                "startup_id": startup_id,
                "startup_name": startup_name,
                "composite_base_value": float(composite_shap.base_values[0]),
                "composite_output_value": float(composite_shap.values.sum() + composite_shap.base_values[0]),
                "pillar_contributions": {
                    pillar: float(composite_shap.values[0][i]) 
                    for i, pillar in enumerate(PILLARS)
                }
            }
            
            # Save as JSON
            import json
            with open(startup_dir / "shap_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            print(f"Error processing startup {row.get('startup_id', i)}: {e}")
    
    print(f"SHAP images and HTML files written to {out_dir}")

if __name__ == "__main__":
    main() 