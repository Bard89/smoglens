import joblib
import pandas as pd
import numpy as np
from pathlib import Path

ensemble_path = Path('/Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained/models_1h.pkl')
print(f"Loading ensemble model from {ensemble_path}")
ensemble_model = joblib.load(ensemble_path)
print(f"Ensemble model type: {type(ensemble_model)}")
print(f"Ensemble model keys: {ensemble_model.keys() if isinstance(ensemble_model, dict) else 'Not a dict'}")

if isinstance(ensemble_model, dict):
    for key, model in ensemble_model.items():
        print(f"\n{key} model:")
        print(f"  Type: {type(model)}")
        if hasattr(model, 'feature_name_'):
            print(f"  Features: {len(model.feature_name_)} features")
            print(f"  First 10 features: {model.feature_name_[:10]}")
        elif hasattr(model, 'feature_names_in_'):
            print(f"  Features: {len(model.feature_names_in_)} features")
            print(f"  First 10 features: {list(model.feature_names_in_[:10])}")

baseline_path = Path('/Users/vojtech/Code/Bard89/smoglens-data/models/baseline_v2_lgb_1h_20250901_024347.pkl')
if baseline_path.exists():
    print(f"\n\nLoading baseline model from {baseline_path}")
    baseline_model = joblib.load(baseline_path)
    print(f"Baseline model type: {type(baseline_model)}")
    if hasattr(baseline_model, 'feature_name_'):
        print(f"Baseline features: {len(baseline_model.feature_name_)} features")
        print(f"First 10 features: {baseline_model.feature_name_[:10]}")

metadata_path = Path('/Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained/metadata.pkl')
if metadata_path.exists():
    print(f"\n\nLoading metadata from {metadata_path}")
    metadata = joblib.load(metadata_path)
    print(f"Metadata keys: {metadata.keys()}")
    if 'feature_columns' in metadata:
        print(f"Feature columns: {len(metadata['feature_columns'])} features")
        print(f"First 10: {metadata['feature_columns'][:10]}")