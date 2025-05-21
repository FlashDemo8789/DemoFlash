#!/bin/bash
# Train hierarchical model architecture for FlashCAMP
# This script trains the pillar models and the meta-model using the hierarchical architecture

set -e # Exit on any error

# Directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if Python environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
  echo "No virtual environment detected. Please activate the virtual environment first."
  echo "Use: source .venv/bin/activate"
  exit 1
fi

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Check for dataset file
DATASET="${PROJECT_ROOT}/flashcamp/data/gold/seed_dataset_master_final_54000_68.csv"
if [ ! -f "$DATASET" ]; then
  echo "Dataset not found at: $DATASET"
  echo "Please make sure the dataset exists at the specified location."
  exit 1
fi

# Create models output directory
V2_DIR="${PROJECT_ROOT}/models/v2"
mkdir -p "${V2_DIR}"

echo "=========================================="
echo "Training Hierarchical Models (v2)"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Output directory: $V2_DIR"
echo "Starting training process..."

# Run the training script
python "${PROJECT_ROOT}/flashcamp/pipelines/train_hierarchical.py" \
  --data "$DATASET" \
  --models "$V2_DIR" \
  --trials 30

# Check if training was successful
if [ $? -eq 0 ]; then
  echo "=========================================="
  echo "Training completed successfully!"
  echo "Models saved to: $V2_DIR"
  echo "=========================================="
  
  # Check if models were created
  if [ -f "${V2_DIR}/success_xgb.joblib" ]; then
    echo "Trained models:"
    ls -lh "${V2_DIR}"
    
    # Create symbolic links in the flashcamp/models directory
    FLASHCAMP_MODELS="${PROJECT_ROOT}/flashcamp/models/v2"
    mkdir -p "${FLASHCAMP_MODELS}"
    
    echo "Creating symbolic links in ${FLASHCAMP_MODELS}..."
    for model in "${V2_DIR}"/*.joblib; do
      filename=$(basename "$model")
      ln -sf "$model" "${FLASHCAMP_MODELS}/$filename"
    done
    
    # Copy metadata
    cp "${V2_DIR}/model_metadata.json" "${FLASHCAMP_MODELS}/"
    
    echo "Symbolic links created."
    echo "You can now use the models for prediction."
  else
    echo "Error: Expected model files not found in ${V2_DIR}"
    exit 1
  fi
else
  echo "Error: Training failed with exit code $?"
  exit 1
fi 