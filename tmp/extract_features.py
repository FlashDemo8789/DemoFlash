import joblib, json, pathlib
model = joblib.load("flashcamp/models/success_xgb.joblib")
names = list(model.get_booster().feature_names)   # -> 79
json.dump(names, open("tmp/model_cols.json", "w"), indent=2)
print(f"Extracted {len(names)} feature names from the model.") 