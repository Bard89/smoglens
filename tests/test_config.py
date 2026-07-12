import json
import pickle
from pathlib import Path

import pytest

from smoglens.config import CACHE_DIR, DATA_ROOT, ENRICHED_DATASET, ENSEMBLE_MODEL_DIR

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_predictions.json"
HORIZONS = ["1h", "2h", "3h", "4h", "5h", "6h", "12h", "24h"]


@pytest.fixture(scope="module")
def data_root_available():
    if not DATA_ROOT.exists():
        pytest.skip("SMOGLENS_DATA_PATH not available")


def test_data_layout_present(data_root_available):
    assert ENRICHED_DATASET.exists()
    assert CACHE_DIR.exists()
    for horizon in HORIZONS:
        assert (ENSEMBLE_MODEL_DIR / f"models_{horizon}.pkl").exists(), horizon


def test_model_metadata_matches_golden_contract(data_root_available):
    with open(ENSEMBLE_MODEL_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert metadata["feature_cols"] == fixture["feature_cols"]
    assert metadata["target_cols"] == [f"target_{h}" for h in HORIZONS]
    assert metadata["categorical_features"] == ["hex_encoded"]
