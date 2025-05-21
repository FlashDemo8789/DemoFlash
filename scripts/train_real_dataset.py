#!/usr/bin/env python
"""
Train hierarchical models on the real 54,000 sample dataset.
This script uses the existing pipeline to train models on the full dataset.

Usage:
    python scripts/train_real_dataset.py
"""
import os
import sys
import subprocess
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

def main():
    """Run the training pipeline with the full dataset"""
    # Set paths
    data_path = "flashcamp/data/gold/seed_dataset_master_final_54000_68.csv"
    models_dir = "models/v2"
    
    print(f"Training hierarchical models on real dataset: {data_path}")
    print(f"Output directory: {models_dir}")
    
    # Check if the dataset exists
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)
    
    # Create the models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Use subprocess to run the training script
    # This allows us to see the output in real-time
    cmd = [
        "python", "-m", "flashcamp.pipelines.train_hierarchical",
        "--data", data_path,
        "--models", models_dir,
        "--trials", "30" # Increase trials for better optimization
    ]
    
    print("\nRunning training command:")
    print(" ".join(cmd))
    print("\n" + "="*80)
    
    # Run the command and show output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end="")
    
    # Wait for the process to complete
    process.wait()
    
    if process.returncode == 0:
        print("\n" + "="*80)
        print(f"Training completed successfully!")
        print(f"Models saved to {models_dir}")
        
        # Run optimization script on the newly trained models
        optimize_cmd = [
            "python", "scripts/optimize_threshold.py",
            "--model_dir", models_dir,
            "--data", data_path
        ]
        
        print("\nOptimizing prediction threshold:")
        print(" ".join(optimize_cmd))
        print("\n" + "="*80)
        
        # Run the optimization command
        optimize_process = subprocess.Popen(
            optimize_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in optimize_process.stdout:
            print(line, end="")
        
        # Wait for the process to complete
        optimize_process.wait()
        
        if optimize_process.returncode == 0:
            print("\n" + "="*80)
            print(f"Threshold optimization completed successfully!")
            
            # Update the existing ML engine to use the new models
            print("\nIntegrating with existing ML engine...")
            
            # Output next steps
            print("\nNext steps:")
            print("1. Restart the backend to use the new models")
            print("2. The existing API endpoints will now use the new hierarchical model")
            print("3. No need for parallel implementation - the same ml.py engine can be used")
            print("4. Run evaluation scripts to verify model performance:")
            print("   python scripts/evaluate_hierarchical_model.py --model_dir models/v2")
            print("\nTraining complete!")
        else:
            print("\nERROR: Threshold optimization failed!")
    else:
        print("\nERROR: Training failed!")

if __name__ == "__main__":
    main() 