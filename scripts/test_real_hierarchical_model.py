#!/usr/bin/env python
"""
Test the trained hierarchical model on the real dataset.
This script loads the models trained on the real dataset and makes predictions.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from flashcamp.features.build_features import build_feature_vector
from flashcamp.pipelines.train_hierarchical import PILLARS

def load_models(model_dir):
    """Load all pillar models and the meta-model"""
    model_dir = Path(model_dir)
    
    # Load pillar models
    pillar_models = {}
    for pillar in PILLARS:
        model_path = model_dir / f"{pillar}_lgbm.joblib"
        if model_path.exists():
            pillar_models[pillar] = joblib.load(model_path)
        else:
            print(f"Warning: {pillar} model not found at {model_path}")
    
    # Load meta-model
    meta_model_path = model_dir / "success_xgb.joblib"
    if meta_model_path.exists():
        meta_model = joblib.load(meta_model_path)
    else:
        print(f"Warning: meta-model not found at {meta_model_path}")
        meta_model = None
    
    # Load feature masks from metadata
    metadata_path = model_dir / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"Warning: metadata not found at {metadata_path}")
        metadata = None
    
    # Load model configuration with optimal threshold if available
    config_path = model_dir / "model_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"Warning: config not found at {config_path}")
        config = None
    
    return pillar_models, meta_model, metadata, config

def predict_startup_success(startup_data, pillar_models, meta_model, metadata, config=None):
    """
    Make predictions for a startup using the hierarchical model.
    
    Args:
        startup_data: Dict containing startup metrics
        pillar_models: Dict of trained pillar models
        meta_model: Trained meta-model
        metadata: Model metadata
        config: Model configuration with optimal threshold
        
    Returns:
        Dict with prediction results
    """
    # Get feature masks from metadata
    feature_masks = {}
    
    # Create a mapping between model feature names and parquet file column names
    feature_mapping = {
        # Capital pillar
        "log_cash_on_hand": "cash_on_hand_usd",
        "ltv_cac_ratio": "ltv_cac_ratio",
        "burn_multiple": "burn_multiple",
        "runway_est": "runway_months",
        "gross_margin": "gross_margin_percent",
        "customer_concentration": "customer_concentration_percent",
        "post_money_valuation": "total_funding_usd",
        
        # Advantage pillar
        "patent_count_norm": "patents_count",
        "network_effect": "has_network_effect",
        "has_data_moat": "has_data_moat",
        "reg_advantage": "regulatory_advantage_present",
        "tech_diff_score": "tech_differentiation_score",
        "switch_cost_score": "switching_cost_score",
        "brand_strength": "brand_strength_score",
        "retention_30d": "product_retention_30d",
        "retention_90d": "product_retention_90d",
        "nps_score_norm": "nps_score",
        
        # Market pillar
        "tam_ratio": "tam_size_usd",
        "sam_ratio": "sam_size_usd",
        "cagr_pct": "claimed_cagr_pct",
        "market_growth_pct": "market_growth_rate_percent",
        "competition_intensity": "competition_intensity",
        "competition_hhi": "top3_competitor_share_pct",
        "reg_level_idx": "industry_regulation_level",
        
        # People pillar
        "founders_count_norm": "founders_count",
        "team_size_norm": "team_size_total",
        "domain_exp_avg": "domain_expertise_years_avg",
        "prior_exits": "previous_exits_count",
        "board_advisor_score": "board_advisor_experience_score",
        "team_diversity": "team_diversity_percent",
        "gender_div_idx": "gender_diversity_index",
        "geo_div_idx": "geography_diversity_index",
        "key_person_dependency": "key_person_dependency"
    }
    
    if metadata and 'pillar_models' in metadata:
        # Map each feature to its column name
        for pillar, pillar_data in metadata['pillar_models'].items():
            feature_list = pillar_data.get('features', [])
            feature_masks[pillar] = []
            for feature in feature_list:
                # Map to the actual column name
                mapped_feature = feature_mapping.get(feature, feature)
                feature_masks[pillar].append(mapped_feature)
    
    # Get threshold from config
    threshold = 0.5
    if config and 'thresholds' in config and 'meta' in config['thresholds']:
        threshold = config['thresholds']['meta']
    
    # Convert input data to feature vector
    features_dict = startup_data
    
    # Make predictions with each pillar model
    pillar_scores = {}
    meta_features = []
    
    for pillar in PILLARS:
        if pillar in pillar_models:
            # Get pillar-specific features if available
            if pillar in feature_masks:
                # Extract the specific features needed for this pillar model
                pillar_feature_names = feature_masks[pillar]
                pillar_feature_values = []
                
                for feature_name in pillar_feature_names:
                    if feature_name in features_dict:
                        value = features_dict[feature_name]
                        # Convert to float if possible
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            # Handle categorical values
                            value = 0.0
                    else:
                        # Feature not found, use a default value
                        print(f"Warning: Feature {feature_name} not found for {pillar}, using 0")
                        value = 0.0
                    
                    pillar_feature_values.append(value)
                
                pillar_features = np.array(pillar_feature_values).reshape(1, -1)
            else:
                # We don't know what features to use - this should not happen
                print(f"ERROR: No feature list for {pillar} in metadata")
                # Just use a neutral value
                pillar_scores[pillar] = 0.5
                meta_features.append(0.5)
                continue
            
            # Make prediction - handle both scikit-learn API and raw LightGBM boosters
            if hasattr(pillar_models[pillar], 'predict_proba'):
                pillar_score = float(pillar_models[pillar].predict_proba(pillar_features)[0, 1])
            else:
                # Raw LightGBM booster
                raw_pred = pillar_models[pillar].predict(pillar_features)
                # Convert to probability
                pillar_score = 1.0 / (1.0 + np.exp(-raw_pred[0]))
            
            pillar_scores[pillar] = pillar_score
            meta_features.append(pillar_score)
        else:
            print(f"Warning: {pillar} model not available, using fallback")
            # Fallback to neutral prediction
            pillar_scores[pillar] = 0.5
            meta_features.append(0.5)
    
    # Reshape for meta-model
    meta_input = np.array(meta_features).reshape(1, -1)
    
    # Make final prediction
    if meta_model:
        if hasattr(meta_model, 'predict_proba'):
            final_score = float(meta_model.predict_proba(meta_input)[0, 1])
        else:
            # Raw XGBoost model
            raw_pred = meta_model.predict(meta_input)
            if len(raw_pred.shape) > 1 and raw_pred.shape[1] > 1:
                # Multi-class output
                final_score = float(raw_pred[0, 1])
            else:
                # Binary output
                final_score = 1.0 / (1.0 + np.exp(-raw_pred[0]))
    else:
        # Fallback if meta-model is not available
        print("Warning: Using fallback meta-prediction")
        pillar_weights = {
            'capital': 0.25,
            'advantage': 0.25,
            'market': 0.25,
            'people': 0.25
        }
        final_score = sum(score * pillar_weights.get(pillar, 0.25) 
                         for pillar, score in pillar_scores.items())
    
    # Return results with threshold
    return {
        'pillar_scores': pillar_scores,
        'final_score': final_score,
        'prediction': 'pass' if final_score >= threshold else 'fail',
        'confidence': max(final_score, 1 - final_score),
        'threshold': threshold
    }

def visualize_prediction(startup_data, prediction_result, output_path=None):
    """Visualize the prediction results"""
    pillars = list(prediction_result['pillar_scores'].keys())
    scores = [prediction_result['pillar_scores'][p] for p in pillars]
    
    # Set up plot
    plt.figure(figsize=(10, 6))
    
    # Create bar chart for pillar scores
    bar_colors = ['#4CAF50' if s >= 0.5 else '#F44336' for s in scores]
    bars = plt.bar(pillars, scores, color=bar_colors, alpha=0.7)
    
    # Add labels and titles
    plt.ylim(0, 1.0)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Add final score indicator
    plt.scatter([len(pillars) - 0.5], [prediction_result['final_score']], 
                color='blue', s=150, zorder=5, label='Final Score')
    
    # Add threshold line
    threshold = prediction_result.get('threshold', 0.5)
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
    plt.text(len(pillars)/2, threshold + 0.02, f'Threshold ({threshold:.2f})', 
             ha='center', color='r')
    
    # Add annotations
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f'{score:.2f}', ha='center')
    
    plt.text(len(pillars) - 0.5, prediction_result['final_score'] + 0.03, 
             f"{prediction_result['final_score']:.2f}", ha='center')
    
    # Add title and styling
    result_text = f"PASS ({prediction_result['confidence']:.0%} confident)" if prediction_result['prediction'] == 'pass' else f"FAIL ({prediction_result['confidence']:.0%} confident)"
    plt.title(f"Startup Success Prediction: {result_text}\nSector: {startup_data.get('sector', 'Unknown')}", fontsize=14)
    plt.ylabel('Success Probability')
    plt.grid(axis='y', alpha=0.3)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the hierarchical model')
    parser.add_argument('--model_dir', type=str, default='models/v2_real', 
                        help='Directory containing the trained models')
    parser.add_argument('--data', type=str, default='flashcamp/data/gold/v1.parquet',
                        help='Path to the dataset')
    parser.add_argument('--sample_id', type=int, default=None,
                        help='Specific sample ID to test (random if not specified)')
    parser.add_argument('--output', type=str, 
                        default='reports/assets/real_prediction_example.png',
                        help='Path to save the visualization')
    
    args = parser.parse_args()
    
    # Load models
    print(f"Loading models from {args.model_dir}...")
    pillar_models, meta_model, metadata, config = load_models(args.model_dir)
    
    # Load dataset
    print(f"Loading dataset from {args.data}...")
    if args.data.endswith('.csv'):
        df = pd.read_csv(args.data)
    else:
        df = pd.read_parquet(args.data)
    
    # Select a sample
    if args.sample_id is not None and args.sample_id < len(df):
        sample = df.iloc[args.sample_id]
    else:
        sample = df.sample(1).iloc[0]
    
    sample_id = sample.name
    print(f"Testing with startup sample ID: {sample_id}")
    
    # Convert sample to dict
    startup_data = sample.to_dict()
    
    # Make prediction
    prediction = predict_startup_success(startup_data, pillar_models, meta_model, metadata, config)
    
    # Print results
    print("\n=== Prediction Results ===")
    print(f"Startup sector: {startup_data.get('sector', 'Unknown')}")
    print(f"Ground truth: {startup_data.get('success_label', 'Unknown')}")
    print(f"Model prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.2f})")
    print(f"Threshold: {prediction['threshold']:.4f}")
    print("\nPillar Scores:")
    for pillar, score in prediction['pillar_scores'].items():
        print(f"  {pillar.upper()}: {score:.4f}")
    print(f"Final Score: {prediction['final_score']:.4f}")
    
    # Visualize
    visualize_prediction(startup_data, prediction, args.output)
    
    # Calculate accuracy on a sample set
    print("\n=== Model Accuracy Test ===")
    test_size = min(100, len(df))
    test_samples = df.sample(test_size, random_state=42)
    
    correct = 0
    for _, row in test_samples.iterrows():
        data = row.to_dict()
        pred = predict_startup_success(data, pillar_models, meta_model, metadata, config)
        
        # Convert true/false to pass/fail for comparison
        true_label = data.get('success_label')
        if isinstance(true_label, bool):
            expected = 'pass' if true_label else 'fail'
        elif isinstance(true_label, str):
            if true_label.lower() in ['true', 'pass', 'success', '1', 'yes']:
                expected = 'pass'
            elif true_label.lower() in ['false', 'fail', 'failure', '0', 'no']:
                expected = 'fail'
            else:
                expected = true_label
        else:
            expected = 'pass' if true_label else 'fail'
            
        if pred['prediction'] == expected:
            correct += 1
    
    accuracy = correct / test_size
    print(f"Tested {test_size} random samples")
    print(f"Accuracy: {accuracy:.2%}") 