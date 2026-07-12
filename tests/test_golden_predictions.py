import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
APP_DIR = REPO_ROOT / "voi" / "V4_streamlit" / "V2_working"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_predictions.json"

sys.path.insert(0, str(APP_DIR))

FIXTURE = json.loads(FIXTURE_PATH.read_text())


def floats_match(expected_repr, actual):
    expected = float(expected_repr)
    if math.isnan(expected) and math.isnan(actual):
        return True
    return math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)


@pytest.fixture(scope="module")
def app_components():
    import config
    from utils.data_processor import DataProcessor
    from utils.feature_generator import FeatureGenerator
    from utils.inference import SimplePredictor

    processed_cache = config.DATA_DIR / "shibuya_processed.parquet"
    if not (Path(config.DATA_PATH).exists() or processed_cache.exists()):
        pytest.skip("local dataset not available")

    predictor = SimplePredictor()
    if not predictor.models:
        pytest.skip("trained models not available")

    data_processor = DataProcessor()
    data_processor.load_data()
    data_processor.find_nearby_hexagons()
    return data_processor, FeatureGenerator(), predictor


@pytest.fixture(scope="module")
def golden_runs(app_components):
    import pandas as pd

    data_processor, feature_generator, predictor = app_components
    runs = {}
    for case in FIXTURE["cases"]:
        timestamp = pd.Timestamp(case["timestamp"], tz="Asia/Tokyo")
        historical_data = data_processor.get_historical_window(timestamp, hours_back=168)
        features_df = feature_generator.generate_features(historical_data)
        last_features = features_df.iloc[-1:].copy()

        model_input = last_features.copy()
        for col in set(predictor.feature_cols) - set(model_input.columns):
            model_input[col] = 0.0
        model_input = model_input[predictor.feature_cols]

        predictions, confidence_intervals = predictor.predict_all(last_features)
        runs[case["timestamp"]] = {
            "n_historical_rows": len(historical_data),
            "model_input": model_input,
            "predictions": predictions,
            "confidence_intervals": confidence_intervals,
        }
    return runs


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["timestamp"])
def test_historical_window_size_unchanged(case, golden_runs):
    assert golden_runs[case["timestamp"]]["n_historical_rows"] == case["n_historical_rows"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["timestamp"])
def test_model_input_features_unchanged(case, golden_runs):
    model_input = golden_runs[case["timestamp"]]["model_input"]
    mismatches = [
        f"{col}: expected {expected_repr}, got {float(model_input.iloc[0][col])!r}"
        for col, expected_repr in case["model_input"].items()
        if not floats_match(expected_repr, float(model_input.iloc[0][col]))
    ]
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["timestamp"])
def test_predictions_unchanged(case, golden_runs):
    predictions = golden_runs[case["timestamp"]]["predictions"]
    for horizon, expected_repr in case["predictions"].items():
        assert floats_match(expected_repr, predictions[horizon]), (
            f"{horizon}: expected {expected_repr}, got {predictions[horizon]!r}"
        )


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["timestamp"])
def test_confidence_intervals_unchanged(case, golden_runs):
    confidence_intervals = golden_runs[case["timestamp"]]["confidence_intervals"]
    for horizon, (expected_lower, expected_upper) in case["confidence_intervals"].items():
        lower, upper = confidence_intervals[horizon]
        assert floats_match(expected_lower, lower), f"{horizon} lower: expected {expected_lower}, got {lower}"
        assert floats_match(expected_upper, upper), f"{horizon} upper: expected {expected_upper}, got {upper}"
