#!/usr/bin/env python
"""
Test the trained hierarchical model by making predictions on sample data.
This script loads the trained models and demonstrates how they work together
to generate predictions.
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
    
    # Load model configuration with optimal threshold
    config_path = model_dir / "model_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"Warning: model config not found at {config_path}")
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
    # Generate feature vector
    feature_vector = build_feature_vector(startup_data)
    
    # Get predictions from pillar models
    pillar_scores = {}
    pillar_features = {}
    for pillar, model in pillar_models.items():
        if metadata and pillar in metadata["pillar_models"]:
            # Get feature indices for this pillar
            feature_names = metadata["pillar_models"][pillar]["features"]
            # Map feature names to indices in FEATURE_COLUMNS
            from flashcamp.features.build_features import FEATURE_COLUMNS
            indices = [FEATURE_COLUMNS.index(name) for name in feature_names]
            
            # Extract features for this pillar
            pillar_features[pillar] = [FEATURE_COLUMNS[i] for i in indices]
            
            # Make prediction
            X_pillar = feature_vector[:, indices]
            pillar_scores[pillar] = float(model.predict(X_pillar)[0])
        else:
            # Fallback if metadata is not available
            print(f"Warning: Using fallback prediction for {pillar} pillar")
            pillar_scores[pillar] = random.random()
    
    # Prepare input for meta-model
    meta_input = np.array([[
        pillar_scores.get('capital', 0.5),
        pillar_scores.get('advantage', 0.5),
        pillar_scores.get('market', 0.5),
        pillar_scores.get('people', 0.5)
    ]])
    
    # Make final prediction
    if meta_model:
        final_score = float(meta_model.predict_proba(meta_input)[0, 1])
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
    
    # Get optimal threshold from config if available
    threshold = 0.5  # Default threshold
    if config and 'thresholds' in config and 'meta' in config['thresholds']:
        threshold = config['thresholds']['meta']
        print(f"Using optimal threshold: {threshold:.4f}")
    
    # Return results
    return {
        'pillar_scores': pillar_scores,
        'final_score': final_score,
        'threshold': threshold,
        'prediction': 'pass' if final_score >= threshold else 'fail',
        'confidence': max(final_score, 1 - final_score)
    }

def visualize_prediction(prediction_result):
    """Create a visualization of the prediction"""
    pillar_scores = prediction_result['pillar_scores']
    
    # Create a bar chart of pillar scores
    plt.figure(figsize=(10, 6))
    
    # Plot pillar scores
    pillars = list(pillar_scores.keys())
    scores = [pillar_scores[p] for p in pillars]
    
    # Define colors based on score
    colors = ['#ff9999' if s < 0.5 else '#99ff99' for s in scores]
    
    # Create bar chart
    bars = plt.bar(pillars, scores, color=colors)
    
    # Add score labels
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{score:.2f}', ha='center', va='bottom')
    
    # Add threshold line
    threshold = prediction_result.get('threshold', 0.5)
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
    plt.text(len(pillars)/2, threshold + 0.02, f'Threshold ({threshold:.4f})', ha='center', color='r')
    
    # Add prediction result
    result_text = f"Final Score: {prediction_result['final_score']:.2f}\nPrediction: {prediction_result['prediction'].upper()}"
    plt.figtext(0.5, 0.01, result_text, ha='center', fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
    
    plt.ylim(0, 1.1)
    plt.title('Startup Success Prediction by Pillar')
    plt.ylabel('Success Score (0-1)')
    plt.tight_layout()
    
    return plt

def main():
    """Main function to test the hierarchical model"""
    import argparse
    parser = argparse.ArgumentParser(description="Test hierarchical model prediction")
    parser.add_argument("--model_dir", default="models/v2", help="Directory with trained models")
    parser.add_argument("--data", default="flashcamp/data/gold/seed_dataset_synthetic.csv", 
                      help="Dataset to sample startups from")
    parser.add_argument("--sample_id", type=int, default=None, 
                      help="ID of sample to test (default: random)")
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
    print("\nPillar Scores:")
    for pillar, score in prediction['pillar_scores'].items():
        print(f"  {pillar.upper()}: {score:.4f}")
    print(f"Final Score: {prediction['final_score']:.4f}")
    print(f"Decision Threshold: {prediction['threshold']:.4f}")
    
    # Create visualization
    plt = visualize_prediction(prediction)
    
    # Save and show plot
    output_dir = Path("reports/assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "prediction_example.png", dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to {output_dir / 'prediction_example.png'}")
    
    # Calculate accuracy on a sample set
    print("\n=== Model Accuracy Test ===")
    test_size = min(100, len(df))
    test_samples = df.sample(test_size, random_state=42)
    
    correct = 0
    for _, row in test_samples.iterrows():
        data = row.to_dict()
        pred = predict_startup_success(data, pillar_models, meta_model, metadata, config)
        if pred['prediction'] == data.get('success_label', 'unknown'):
            correct += 1
    
    accuracy = correct / test_size
    print(f"Tested {test_size} random samples")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main() 