#!/usr/bin/env python
"""
Evaluate the trained hierarchical model's performance on the test set.
This script performs a detailed evaluation of the model, including:
- Overall accuracy, precision, recall, and F1 score
- Comparison of pillar model performance
- Feature importance analysis
- Confusion matrix visualization
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from flashcamp.features.build_features import build_feature_vector, FEATURE_COLUMNS
from flashcamp.pipelines.train_hierarchical import PILLARS, create_feature_masks

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
        
    # Load metadata
    metadata_path = model_dir / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"Warning: metadata not found at {metadata_path}")
        metadata = None
    
    return pillar_models, meta_model, metadata

def get_pillar_feature_indices(metadata):
    """Get indices of features for each pillar from metadata"""
    pillar_indices = {}
    
    for pillar in PILLARS:
        if pillar in metadata["pillar_models"]:
            # Get feature names for this pillar
            feature_names = metadata["pillar_models"][pillar]["features"]
            # Map feature names to indices in FEATURE_COLUMNS
            indices = [FEATURE_COLUMNS.index(name) for name in feature_names]
            pillar_indices[pillar] = indices
    
    return pillar_indices

def predict_batch(X, pillar_models, meta_model, pillar_indices):
    """Make predictions for a batch of samples using pillar models and meta-model"""
    n_samples = X.shape[0]
    
    # Get predictions from pillar models
    pillar_preds = np.zeros((n_samples, len(PILLARS)))
    
    for i, pillar in enumerate(PILLARS):
        if pillar in pillar_models and pillar in pillar_indices:
            indices = pillar_indices[pillar]
            pillar_preds[:, i] = pillar_models[pillar].predict(X[:, indices])
    
    # Get meta-model predictions
    if meta_model:
        meta_preds = meta_model.predict_proba(pillar_preds)[:, 1]
    else:
        # Simple average if no meta-model
        meta_preds = np.mean(pillar_preds, axis=1)
    
    # Binary predictions
    binary_preds = (meta_preds >= 0.5).astype(int)
    
    return meta_preds, binary_preds, pillar_preds

def evaluate_model_performance(y_true, y_pred, y_prob):
    """Calculate and return performance metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate AUC if we have probability predictions
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5  # Default for failed AUC calculation
    else:
        auc = None
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Success'],
                yticklabels=['Fail', 'Success'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_pillar_performance(pillar_metrics, output_dir):
    """Plot and save pillar performance comparison"""
    pillars = list(pillar_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        values = [pillar_metrics[p][metric] for p in pillars]
        bars = plt.bar(pillars, values)
        
        # Add value labels
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.02,
                    f'{val:.2f}', 
                    ha='center', va='bottom')
        
        plt.title(f'{metric.upper()}')
        plt.ylim(0, 1.1)
        
    plt.tight_layout()
    plt.savefig(output_dir / "pillar_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_score_distributions(pillar_scores, y_true, output_dir):
    """Plot and save distributions of scores by true label"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Convert to pandas for easier plotting
    scores_df = pd.DataFrame(pillar_scores, columns=PILLARS)
    scores_df['meta'] = np.mean(scores_df[list(PILLARS)], axis=1)  # Simplified meta score
    scores_df['label'] = y_true
    
    # Plot distributions for each pillar and meta-model
    for i, column in enumerate(PILLARS + ['meta']):
        if i < len(axes):
            ax = axes[i]
            # Plot for failures
            sns.kdeplot(scores_df[scores_df['label'] == 0][column], 
                        ax=ax, label='Fail', color='red', fill=True, alpha=0.3)
            
            # Plot for successes
            sns.kdeplot(scores_df[scores_df['label'] == 1][column], 
                        ax=ax, label='Success', color='green', fill=True, alpha=0.3)
            
            ax.set_title(f'{column.upper()} Score Distribution')
            ax.set_xlim(0, 1)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    """Main function to evaluate model performance"""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate hierarchical model performance")
    parser.add_argument("--model_dir", default="models/v2", help="Directory with trained models")
    parser.add_argument("--data", default="flashcamp/data/gold/seed_dataset_synthetic.csv", 
                      help="Dataset to evaluate on")
    parser.add_argument("--output_dir", default="reports/assets/evaluation",
                      help="Directory to save evaluation results")
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
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Get feature indices for each pillar
    pillar_indices = get_pillar_feature_indices(metadata) if metadata else {}
    
    # Make predictions with each pillar model and the meta-model
    print("\nMaking predictions...")
    meta_probs, meta_preds, pillar_scores = predict_batch(X, pillar_models, meta_model, pillar_indices)
    
    # Evaluate overall model performance
    print("\n=== Overall Model Performance ===")
    meta_metrics = evaluate_model_performance(y, meta_preds, meta_probs)
    for metric, value in meta_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Save classification report
    class_report = classification_report(y, meta_preds, target_names=['Fail', 'Success'])
    print(f"\nClassification Report:\n{class_report}")
    
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(class_report)
    
    # Evaluate each pillar model separately
    print("\n=== Pillar Model Performance ===")
    pillar_metrics = {}
    
    for i, pillar in enumerate(PILLARS):
        pillar_probs = pillar_scores[:, i]
        pillar_preds = (pillar_probs >= 0.5).astype(int)
        
        metrics = evaluate_model_performance(y, pillar_preds, pillar_probs)
        pillar_metrics[pillar] = metrics
        
        print(f"\n{pillar.upper()} Pillar:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Confusion matrix
    plot_confusion_matrix(y, meta_preds, output_dir)
    
    # 2. Pillar performance comparison
    plot_pillar_performance(pillar_metrics, output_dir)
    
    # 3. Score distributions
    plot_score_distributions(pillar_scores, y, output_dir)
    
    # Save evaluation results to JSON
    results = {
        "meta_model": meta_metrics,
        "pillar_models": pillar_metrics,
        "dataset": {
            "path": args.data,
            "size": len(df),
            "class_distribution": {
                "fail": int(np.sum(y == 0)),
                "pass": int(np.sum(y == 1))
            }
        }
    }
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to {output_dir}")

if __name__ == "__main__":
    main() 