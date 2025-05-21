#!/usr/bin/env python
"""
Test the enhanced ML engine with integrated hierarchical model functionality.
This script verifies that the enhanced ML engine works correctly with the
models trained on the real dataset.
"""
import sys
import os
import json
from pathlib import Path
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from flashcamp.backend.app.engines.ml import (
    predict_success_probability,
    generate_recommendations,
    get_model_metadata
)

def load_sample_data(data_path=None):
    """Load sample data from the real dataset"""
    if data_path is None:
        data_path = "flashcamp/data/gold/seed_dataset_master_final_54000_68.csv"
        if not os.path.exists(data_path):
            data_path = "flashcamp/data/gold/v1.parquet"
    
    print(f"Loading sample data from {data_path}")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)
    
    # Sample a few startups randomly
    sample_ids = random.sample(range(len(df)), 5)
    samples = [df.iloc[i].to_dict() for i in sample_ids]
    
    return samples

def test_prediction(startup_data):
    """Test the prediction functionality"""
    print("\n=== Testing Prediction ===")
    
    result = predict_success_probability(startup_data)
    
    print(f"Startup sector: {startup_data.get('sector', 'Unknown')}")
    print(f"Success prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
    print(f"Final score: {result['final_score']:.4f} (Threshold: {result['threshold']:.4f})")
    print("Pillar scores:")
    for pillar, score in result['pillar_scores'].items():
        print(f"  {pillar.capitalize()}: {score:.4f}")
    
    return result

def test_recommendations(startup_data):
    """Test the recommendation functionality"""
    print("\n=== Testing Recommendations ===")
    
    recommendations = generate_recommendations(startup_data)
    
    for pillar, recs in recommendations.items():
        if recs:
            print(f"\n{pillar.upper()} PILLAR RECOMMENDATIONS:")
            for rec in recs:
                print(f"  • {rec['recommendation']} (Impact: {rec['impact']})")
        else:
            print(f"\n{pillar.upper()} PILLAR: No recommendations")
    
    return recommendations

def test_model_metadata():
    """Test the model metadata functionality"""
    print("\n=== Testing Model Metadata ===")
    
    metadata = get_model_metadata()
    
    print(f"Model version: {metadata['model_version']}")
    print(f"Dataset size: {metadata['dataset_size']}")
    print(f"Success rate: {metadata.get('success_rate', 0):.2%}")
    print(f"Threshold: {metadata['threshold']:.4f}")
    
    if metadata.get('meta_metrics'):
        print("\nMeta-model metrics:")
        for metric, value in metadata['meta_metrics'].items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
    
    return metadata

def visualize_prediction(result, startup_data, output_path='reports/assets/enhanced_ml_test.png'):
    """Visualize the prediction result"""
    print("\n=== Creating Visualization ===")
    
    # Extract data from prediction result
    pillars = list(result.get('pillar_scores', {}).keys())
    scores = [result.get('pillar_scores', {}).get(p, 0.5) for p in pillars]
    final_score = result.get('final_score', 0.5)
    threshold = result.get('threshold', 0.5)
    prediction = result.get('prediction', 'unknown')
    confidence = result.get('confidence', 0.5)
    
    # Create figure and axes
    plt.figure(figsize=(10, 6))
    
    # Create bar chart for pillar scores
    bar_colors = ['#4CAF50' if s >= 0.5 else '#F44336' for s in scores]
    bars = plt.bar(pillars, scores, color=bar_colors, alpha=0.7)
    
    # Add labels and titles
    plt.ylim(0, 1.0)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Add final score indicator
    plt.scatter([len(pillars) - 0.5], [final_score], 
                color='blue', s=150, zorder=5, label='Final Score')
    
    # Add threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
    plt.text(len(pillars)/2, threshold + 0.02, f'Threshold ({threshold:.2f})', 
             ha='center', color='r')
    
    # Add annotations
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f'{score:.2f}', ha='center')
    
    plt.text(len(pillars) - 0.5, final_score + 0.03, 
             f"{final_score:.2f}", ha='center')
    
    # Add title and styling
    result_text = f"PASS ({confidence:.0%} confident)" if prediction == 'pass' else f"FAIL ({confidence:.0%} confident)"
    plt.title(f"Enhanced ML Engine Test: {result_text}\nSector: {startup_data.get('sector', 'Unknown')}", fontsize=14)
    plt.ylabel('Success Probability')
    plt.grid(axis='y', alpha=0.3)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def main():
    """Main test function"""
    print("=== Testing Enhanced ML Engine with Real Dataset ===")
    print("Using models trained on 54,000 samples in models/v2/")
    
    # Load sample data
    samples = load_sample_data()
    
    # Test the first sample
    sample = samples[0]
    
    # Test the prediction functionality
    result = test_prediction(sample)
    
    # Test the recommendation functionality
    recommendations = test_recommendations(sample)
    
    # Test the model metadata functionality
    metadata = test_model_metadata()
    
    # Create visualization
    visualize_prediction(result, sample)
    
    # Test a few more samples
    print("\n=== Testing Multiple Samples ===")
    for i, sample in enumerate(samples[1:3], 1):
        print(f"\nTesting sample {i+1}:")
        result = test_prediction(sample)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 