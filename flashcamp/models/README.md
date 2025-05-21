# FlashCAMP ML Models

This directory contains the machine learning models used by FlashCAMP to predict startup success probability.

## Model Architecture

FlashCAMP uses a hierarchical ensemble model architecture:

1. **Pillar Models**: Four specialized LightGBM models, each focused on a different aspect of startup success:
   - `capital_lgbm.joblib`: Financial and fundraising metrics
   - `advantage_lgbm.joblib`: Competitive moat and product metrics 
   - `market_lgbm.joblib`: Market size, growth, and competition metrics
   - `people_lgbm.joblib`: Team composition and experience metrics

2. **Composite Model**: An XGBoost model (`success_xgb.joblib`) that combines the outputs of the pillar models to make the final prediction.

## Model Versioning

Models are stored in versioned directories:
- `/v1/`: Original models (if available)
- `/v2/`: Current models trained on the 54,000 startup dataset

## Retraining the Models

### Prerequisites

1. Make sure you have the dataset available at `flashcamp/data/seed_dataset_master_final_54000_68.csv`
2. Ensure all dependencies are installed: `pip install -r requirements.txt`

### Training Process

To retrain the models, run the training script:

```bash
./flashcamp/scripts/train_models.sh
```

This script will:
1. Train all pillar models using Optuna for hyperparameter optimization
2. Train the composite model
3. Generate SHAP explanation plots for a sample of startups
4. Save all models and metadata to the `models/v2/` directory

### Customizing Training

You can modify the training parameters in `flashcamp/pipelines/train_baseline.py`:

- Adjust the number of Optuna trials for hyperparameter search
- Change cross-validation settings
- Modify the feature selection for each pillar

## Model Evaluation

After training, you can evaluate the models using:

```bash
python flashcamp/pipelines/evaluate_models.py --models models/v2/
```

This will generate performance metrics and validation plots.

## Adding to Production

To use the new models in production:

1. Copy the trained models to the `models/v2/` directory in your production environment
2. The API will automatically use the new models for predictions

## Model Metadata

Each model version includes a `model_metadata.json` file with:
- Training dataset details
- Model hyperparameters
- Performance metrics
- Training date

## SHAP Explanations

SHAP (SHapley Additive exPlanations) values are used to explain model predictions.

SHAP visualizations for sample startups are generated during training and stored in `flashcamp/reports/assets/shap_v2/`. 