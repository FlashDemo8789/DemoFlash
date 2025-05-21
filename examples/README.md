# FlashCAMP Examples

This directory contains example scripts demonstrating how to use the FlashCAMP platform and its features.

## Available Examples

### Hierarchical Model Usage

The `hierarchical_model_usage.py` script demonstrates how to use the hierarchical model architecture for startup success prediction.

Features demonstrated:
- Making predictions with the hierarchical model
- Analyzing pillar scores and contributions
- Generating recommendations based on pillar strengths/weaknesses
- Visualizing pillar scores

To run the example:

```bash
cd examples
python hierarchical_model_usage.py
```

This will:
1. Load the hierarchical model (or use the fallback method if models aren't found)
2. Analyze sample startup data
3. Output detailed analysis including overall success probability and pillar scores
4. Generate pillar score visualizations
5. Provide targeted recommendations based on the weakest pillar

## Using Your Own Data

You can modify the `startup_data` dictionary in any example script to test with your own metrics:

```python
startup_data = {
    "cash_on_hand_usd": 5000000,
    "runway_months": 18,
    # Add other metrics...
}
```

For a complete list of available metrics, see the frontend metrics configuration file at `flashcamp/frontend/constants/metrics.json`.

## Training Custom Models

To train custom hierarchical models with your own data, use the training script:

```bash
cd FLASH
./scripts/train_hierarchical_models.sh
```

This will train the four pillar models and the meta-model, and save them to the `models/v2/` directory. 