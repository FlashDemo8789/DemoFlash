#!/usr/bin/env python
"""
Optimize the prediction threshold for the hierarchical model.
This script tests different threshold values for converting 
probability scores to binary decisions, helping to balance 
precision and recall.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc
)

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from flashcamp.features.build_features import build_feature_vector, FEATURE_COLUMNS
from flashcamp.pipelines.train_hierarchical import PILLARS
from scripts.evaluate_hierarchical_model import (
    load_models, get_pillar_feature_indices, predict_batch
)

def find_optimal_threshold(y_true, y_probs, metric="f1"):
    """Find optimal threshold based on different metrics"""
    # Calculate precision, recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Calculate F1 for each threshold
    f1_scores = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        preds = (y_probs >= threshold).astype(int)
        f1_scores[i] = f1_score(y_true, preds)
    
    # Find best threshold based on metric
    if metric == "f1":
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_score = f1_scores[best_idx]
    elif metric == "precision_recall_balance":
        # Find point where precision and recall are balanced
        best_idx = np.argmin(np.abs(precision - recall))
        best_threshold = thresholds[best_idx]
        best_score = (precision[best_idx] + recall[best_idx]) / 2
    
    return best_threshold, best_score

def plot_threshold_metrics(y_true, y_probs, output_dir):
    """Plot accuracy, precision, recall, and F1 for different thresholds"""
    thresholds = np.linspace(0.01, 0.99, 99)
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        metrics["accuracy"].append(accuracy_score(y_true, y_pred))
        # Handle case where no positive predictions
        try:
            metrics["precision"].append(precision_score(y_true, y_pred))
        except:
            metrics["precision"].append(0)
        metrics["recall"].append(recall_score(y_true, y_pred))
        # Handle case where precision or recall is 0
        try:
            metrics["f1"].append(f1_score(y_true, y_pred))
        except:
            metrics["f1"].append(0)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    for metric_name, values in metrics.items():
        plt.plot(thresholds, values, label=metric_name)
    
    # Find optimal F1 threshold
    best_threshold, best_f1 = find_optimal_threshold(y_true, y_probs, "f1")
    plt.axvline(x=best_threshold, color='red', linestyle='--', 
                label=f'Best Threshold = {best_threshold:.2f} (F1 = {best_f1:.2f})')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "threshold_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return best_threshold, best_f1

def plot_roc_curve(y_true, y_probs, output_dir):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_precision_recall_curve(y_true, y_probs, output_dir):
    """Plot precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    
    # Find threshold with balanced precision-recall
    best_threshold, best_score = find_optimal_threshold(y_true, y_probs, "precision_recall_balance")
    
    # Find index of best threshold in thresholds
    best_idx = np.argmin(np.abs(thresholds - best_threshold))
    
    plt.scatter(recall[best_idx], precision[best_idx], marker='o', color='red', s=100,
               label=f'Balanced Point (threshold={best_threshold:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return best_threshold

def evaluate_with_threshold(y_true, y_prob, threshold):
    """Evaluate model performance with a specific threshold"""
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold
    }
    
    return metrics

def main():
    """Main function to find optimal threshold"""
    import argparse
    parser = argparse.ArgumentParser(description="Optimize prediction threshold")
    parser.add_argument("--model_dir", default="models/v2", help="Directory with trained models")
    parser.add_argument("--data", default="flashcamp/data/gold/seed_dataset_synthetic.csv", 
                      help="Dataset to evaluate on")
    parser.add_argument("--output_dir", default="reports/assets/threshold_optimization",
                      help="Directory to save results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print(f"Loading models from {args.model_dir}...")
    pillar_models, meta_model, metadata = load_models(args.model_dir)
    
    # Load dataset
    print(f"Loading dataset from {args.data}...")
    if args.data.endswith('.csv'):
        df = pd.read_csv(args.data)
    else:
        df = pd.read_parquet(args.data)
    
    print(f"Dataset loaded with {len(df)} samples")
    
    # Process samples
    X_list = []
    y_list = []
    
    # Convert success_label to binary
    label_map = {"pass": 1, "fail": 0}
    
    # Process each row to extract features
    print("Extracting features...")
    for _, row in df.iterrows():
        try:
            X_list.append(build_feature_vector(row.to_dict()))
            
            # Get label
            label = row.get("success_label")
            if isinstance(label, str):
                y_list.append(label_map.get(label, 0))
            else:
                y_list.append(int(label > 0))
                
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    # Stack features and labels
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    # Get feature indices for each pillar
    pillar_indices = get_pillar_feature_indices(metadata) if metadata else {}
    
    # Make predictions
    print("\nMaking predictions...")
    meta_probs, meta_preds_default, pillar_scores = predict_batch(X, pillar_models, meta_model, pillar_indices)
    
    # Default threshold performance (0.5)
    default_metrics = evaluate_with_threshold(y, meta_probs, 0.5)
    print("\n=== Default Threshold (0.5) ===")
    for metric, value in default_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Find and evaluate with optimal threshold for F1
    print("\n=== Finding Optimal Threshold ===")
    optimal_threshold, optimal_f1 = plot_threshold_metrics(y, meta_probs, output_dir)
    
    # Evaluate with optimal threshold
    optimal_metrics = evaluate_with_threshold(y, meta_probs, optimal_threshold)
    print(f"\n=== Optimal Threshold ({optimal_threshold:.4f}) ===")
    for metric, value in optimal_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Plot ROC curve
    plot_roc_curve(y, meta_probs, output_dir)
    
    # Plot precision-recall curve and find balanced threshold
    balanced_threshold = plot_precision_recall_curve(y, meta_probs, output_dir)
    
    # Evaluate with balanced threshold
    balanced_metrics = evaluate_with_threshold(y, meta_probs, balanced_threshold)
    print(f"\n=== Balanced Threshold ({balanced_threshold:.4f}) ===")
    for metric, value in balanced_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Try a range of manual thresholds
    print("\n=== Testing Manual Thresholds ===")
    manual_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    manual_results = {}
    
    for threshold in manual_thresholds:
        metrics = evaluate_with_threshold(y, meta_probs, threshold)
        manual_results[threshold] = metrics
        print(f"\nThreshold = {threshold:.1f}")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    # Save results
    # Convert any numpy types to Python native types for JSON serialization
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(i) for i in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    results = {
        "default_metrics": default_metrics,
        "optimal_metrics": optimal_metrics,
        "balanced_metrics": balanced_metrics,
        "manual_thresholds": {str(k): v for k, v in manual_results.items()},
        "dataset": {
            "path": args.data,
            "size": len(df),
            "positive_samples": int(np.sum(y == 1)),
            "negative_samples": int(np.sum(y == 0))
        },
        "meta_model_range": {
            "min": float(np.min(meta_probs)),
            "max": float(np.max(meta_probs)),
            "mean": float(np.mean(meta_probs)),
            "median": float(np.median(meta_probs)),
            "std": float(np.std(meta_probs))
        }
    }
    
    # Convert any numpy types to native Python types
    results = convert_to_native_types(results)
    
    with open(output_dir / "threshold_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Create a model config with optimal threshold
    model_config = {
        "pillars": PILLARS,
        "thresholds": {
            "meta": float(optimal_threshold),
            "balanced": float(balanced_threshold)
        },
        "feature_columns": FEATURE_COLUMNS,
        "training_data": {
            "path": args.data,
            "size": len(df),
            "class_distribution": {
                "fail": int(np.sum(y == 0)),
                "pass": int(np.sum(y == 1))
            }
        },
        "performance": {
            "default": default_metrics,
            "optimal": optimal_metrics,
            "balanced": balanced_metrics
        }
    }
    
    # Convert any numpy types to native Python types
    model_config = convert_to_native_types(model_config)
    
    with open(Path(args.model_dir) / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Model config with optimal threshold saved to {Path(args.model_dir) / 'model_config.json'}")

if __name__ == "__main__":
    main() 