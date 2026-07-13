import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from smoglens.config import ENRICHED_DATASET, ENSEMBLE_MODEL_DIR
from smoglens.data import DataProcessor
from smoglens.features import FeatureGenerator
from smoglens.inference import SimplePredictor

REPO_ROOT = Path(__file__).parents[2]
APP_CACHE_DIR = REPO_ROOT / "apps" / "streamlit" / "data"
FIXTURES_DIR = Path(__file__).parent

TIMESTAMPS = [
    "2024-06-01 12:00:00",
    "2024-11-03 22:00:00",
    "2025-01-15 08:00:00",
]


def build_golden_cases():
    data_processor = DataProcessor(data_path=ENRICHED_DATASET, cache_dir=APP_CACHE_DIR)
    data_processor.load_data()
    data_processor.find_nearby_hexagons()

    feature_generator = FeatureGenerator()
    predictor = SimplePredictor(model_dir=ENSEMBLE_MODEL_DIR)

    cases = []
    for ts in TIMESTAMPS:
        timestamp = pd.Timestamp(ts, tz="Asia/Tokyo")
        historical_data = data_processor.get_historical_window(timestamp, hours_back=168)
        features_df = feature_generator.generate_features(historical_data)
        last_features = features_df.iloc[-1:].copy()

        model_input = last_features.copy()
        for col in set(predictor.feature_cols) - set(model_input.columns):
            model_input[col] = 0.0
        model_input = model_input[predictor.feature_cols]

        predictions, confidence_intervals = predictor.predict_all(last_features)

        cases.append(
            {
                "timestamp": ts,
                "n_historical_rows": len(historical_data),
                "current_pm25": repr(float(data_processor.get_data_at_time(timestamp)["pm25"])),
                "model_input": {col: repr(float(model_input.iloc[0][col])) for col in predictor.feature_cols},
                "predictions": {h: repr(float(p)) for h, p in predictions.items()},
                "confidence_intervals": {
                    h: [repr(float(ci[0])), repr(float(ci[1]))] for h, ci in confidence_intervals.items()
                },
            }
        )

    return {
        "description": "Golden values pinning the streamlit app behavior during the cleanup refactor",
        "environment": {
            "python": platform.python_version(),
            **{lib: version(lib) for lib in ["numpy", "pandas", "lightgbm", "xgboost", "catboost"]},
        },
        "feature_cols": predictor.feature_cols,
        "cases": cases,
    }


def build_model_manifest():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ENSEMBLE_MODEL_DIR.glob("*.pkl"))
    }


def main():
    golden_path = FIXTURES_DIR / "golden_predictions.json"
    manifest_path = FIXTURES_DIR / "model_manifest.json"

    golden_path.write_text(json.dumps(build_golden_cases(), indent=2) + "\n")
    manifest_path.write_text(json.dumps(build_model_manifest(), indent=2) + "\n")
    print(f"wrote {golden_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
