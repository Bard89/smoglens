import joblib
import pandas as pd
from pathlib import Path

metadata_path = Path('/Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained/metadata.pkl')
metadata = joblib.load(metadata_path)

print("Feature columns from metadata:")
print(f"Number of features: {len(metadata['feature_cols'])}")
print("\nAll features:")
for i, feat in enumerate(metadata['feature_cols']):
    print(f"{i+1:2d}. {feat}")

print(f"\n\nTarget columns: {metadata['target_cols']}")
print(f"Categorical features: {metadata['categorical_features']}")

data_path = Path('data/shibuya_optimal_period.csv.gz')
df = pd.read_csv(data_path, compression='gzip', nrows=5)
print(f"\n\nColumns in our dataset ({len(df.columns)} total):")
for col in sorted(df.columns):
    print(f"  - {col}")