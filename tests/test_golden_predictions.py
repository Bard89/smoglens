import json
import math
from pathlib import Path

import pandas as pd
import pytest

from smoglens.config import ENRICHED_DATASET, ENSEMBLE_MODEL_DIR
from smoglens.data import DataProcessor
from smoglens.features import FeatureGenerator
from smoglens.inference import SimplePredictor

REPO_ROOT = Path(__file__).parents[1]
APP_CACHE_DIR = REPO_ROOT / "apps" / "streamlit" / "data"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_predictions.json"

FIXTURE = json.loads(FIXTURE_PATH.read_text())


def floats_match(expected_repr, actual):
    expected = float(expected_repr)
    if math.isnan(expected) and math.isnan(actual):
        return True
    return math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)


@pytest.fixture(scope="module")
def app_components():
    data_processor = DataProcessor(data_path=ENRICHED_DATASET, cache_dir=APP_CACHE_DIR)
    if not (ENRICHED_DATASET.exists() or data_processor.cache_path.exists()):
        pytest.skip("local dataset not available")
    if not (ENSEMBLE_MODEL_DIR / "models_1h.pkl").exists():
        pytest.skip("trained models not available")

    predictor = SimplePredictor(model_dir=ENSEMBLE_MODEL_DIR)
    data_processor.load_data()
    data_processor.find_nearby_hexagons()
    return data_processor, FeatureGenerator(), predictor


@pytest.fixture(scope="module")
def golden_runs(app_components):
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
            "current_pm25": float(data_processor.get_data_at_time(timestamp)["pm25"]),
            "model_input": model_input,
            "predictions": predictions,
            "confidence_intervals": confidence_intervals,
        }
    return runs


def test_feature_columns_unchanged(app_components):
    _, _, predictor = app_components
    assert predictor.feature_cols == FIXTURE["feature_cols"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["timestamp"])
def test_historical_window_size_unchanged(case, golden_runs):
    assert golden_runs[case["timestamp"]]["n_historical_rows"] == case["n_historical_rows"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["timestamp"])
def test_current_pm25_unchanged(case, golden_runs):
    actual = golden_runs[case["timestamp"]]["current_pm25"]
    assert floats_match(case["current_pm25"], actual), f"expected {case['current_pm25']}, got {actual}"


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
