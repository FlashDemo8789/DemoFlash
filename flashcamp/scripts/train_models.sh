#!/bin/bash
# Model Training Script for FlashCAMP V2 Models
# This script runs the full model training pipeline from the new dataset

set -e  # Exit on error

echo "Starting FlashCAMP V2 model training pipeline..."

# Create directory structure if it doesn't exist
mkdir -p models/v2
mkdir -p flashcamp/reports/assets/shap_v2

# Step 1: Train the models
echo "Training baseline pillar and composite models..."
python flashcamp/pipelines/train_baseline.py \
  --data flashcamp/data/seed_dataset_master_final_54000_68.csv \
  --models models/v2 \
  --sample 5000  # Use 5000 samples to speed up training for testing

# Step 2: Generate SHAP values for a sample of startups
echo "Generating SHAP explanations for sample startups..."
python flashcamp/pipelines/gen_shap.py \
  --data flashcamp/data/seed_dataset_master_final_54000_68.csv \
  --models models/v2 \
  --out flashcamp/reports/assets/shap_v2 \
  --sample 100  # Generate SHAP for 100 startups

echo "Model training pipeline complete!"
echo "Models saved to: models/v2/"
echo "SHAP explanations saved to: flashcamp/reports/assets/shap_v2/" 