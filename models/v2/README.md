# FlashCAMP Hierarchical Model - Version 2

This directory contains the trained hierarchical model files and metadata for the FlashCAMP startup success prediction system.

## Model Overview

The hierarchical model architecture consists of:

1. Four pillar models (LightGBM classifiers):
   - `capital_lgbm.joblib`: Financial and funding metrics
   - `advantage_lgbm.joblib`: Competitive advantages and product metrics
   - `market_lgbm.joblib`: Market size and competition metrics
   - `people_lgbm.joblib`: Team composition and experience metrics

2. Meta-model (XGBoost classifier):
   - `success_xgb.joblib`: Combines pillar model outputs for final prediction

## Key Files

- `model_metadata.json`: Details about model training, hyperparameters, and dataset
- `model_config.json`: Runtime configuration including optimized thresholds
- `feature_importance.json`: SHAP-based feature importance for each model
- `shap_values/`: Directory containing SHAP value matrices for visualization

## Performance Metrics

- **Capital Pillar Model**: AUC = 0.7527, Accuracy = 76.4%
- **Advantage Pillar Model**: AUC = 0.5135, Accuracy = 72.4%
- **Market Pillar Model**: AUC = 0.5387, Accuracy = 72.4%
- **People Pillar Model**: AUC = 0.5346, Accuracy = 72.4%
- **Meta-Model**: AUC = 0.7508, Accuracy = 79.2%
- **Optimized Threshold**: 0.304 (F1-optimized)

## Dataset Information

- 54,000 startup samples with known outcomes
- 60 features across all pillars
- 27.6% success rate (balanced through stratified sampling)

## Frontend Integration

The model is integrated into the FlashCAMP frontend through:

1. HierarchicalModelResults component
2. RecommendationsPanel component
3. PredictionVisualization component
4. ModelInfoPanel component

These components connect to the backend API endpoints in `/api/prediction/`.

## Usage

To use the hierarchical model for prediction:

```python
from flashcamp.backend.app.engines.ml import predict_success_probability

# Input metrics dictionary
metrics = {
    "runway_months": 12,
    "burn_multiple": 1.5,
    # ... other metrics
}

# Get prediction
result = predict_success_probability(metrics)
print(f"Success Probability: {result['success_probability']}")
print(f"Pillar Scores: {result['pillar_scores']}")
```

## Future Improvements

1. Further optimization of thresholds for specific use cases
2. Ensemble models for each pillar to improve robustness
3. Implementing online learning for continuous model improvement
4. Adding sector-specific models for improved accuracy

## Contributors

- FlashCAMP Data Science Team
- FlashCAMP Frontend Engineering Team

## Last Updated

July 18, 2024 